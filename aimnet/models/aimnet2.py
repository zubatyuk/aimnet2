import torch
from torch import nn, Tensor
from typing import Dict, List, Tuple, Union
from aimnet import ops, nbops
from aimnet.aev import AEVSV, ConvSV
from aimnet.modules import MLP, Embedding
from aimnet.models.base import AIMNet2Base


class AIMNet2(AIMNet2Base):
    def __init__(self, aev: Dict, nfeature: int, d2features: bool, ncomb_v: int, hidden: Tuple[List[int]],
                 aim_size: int, outputs: Union[List[nn.Module], Dict[str, nn.Module]],
                 num_charge_channels: int = 1):
        super().__init__()

        self.aev = AEVSV(**aev)
        nshifts_s = aev['nshifts_s']
        nshifts_v = aev.get('nshitfs_v') or nshifts_s
        if d2features:
            assert nshifts_s == nshifts_v
            nfeature_tot = nshifts_s * nfeature
        else:
            nfeature_tot = nfeature
        self.nfeature = nfeature
        self.nshifts_s = nshifts_s
        self.d2features = d2features

        self.afv = Embedding(num_embeddings=64, embedding_dim=nfeature, padding_idx=0)

        with torch.no_grad():
            nn.init.orthogonal_(self.afv.weight[1:])
            if d2features:
                self.afv.weight = nn.Parameter(self.afv.weight.clone().unsqueeze(-1).expand(64, nfeature, nshifts_s).flatten(-2, -1))

        conv_param = dict(nshifts_s=nshifts_s, nshifts_v=nshifts_v,
                          ncomb_v=ncomb_v, do_vector=True)
        self.conv_a = ConvSV(nchannel=nfeature, d2features=d2features, **conv_param)
        self.conv_q = ConvSV(nchannel=1, d2features=False, **conv_param)

        mlp_param = {'activation_fn': nn.GELU(), 'last_linear': True}
        mlps = [MLP(n_in=self.conv_a.output_size() + nfeature_tot,
                   n_out=nfeature_tot+2*num_charge_channels, hidden=hidden[0], **mlp_param)]
        mlp_param = {'activation_fn': nn.GELU(), 'last_linear': False}
        for h in hidden[1:-1]:
            mlps.append(MLP(n_in=self.conv_a.output_size() + self.conv_q.output_size() +
                   nfeature_tot+num_charge_channels, n_out=nfeature_tot+2*num_charge_channels, hidden=h, **mlp_param))
        mlp_param = {'activation_fn': nn.GELU(), 'last_linear': False}
        mlps.append(MLP(n_in=self.conv_a.output_size() + self.conv_q.output_size() +
                   nfeature_tot+num_charge_channels, n_out=aim_size, hidden=hidden[-1], **mlp_param))
        self.mlps = nn.ModuleList(mlps)

        if isinstance(outputs, list):
            self.outputs = nn.ModuleList(outputs)
        elif isinstance(outputs, dict):
            self.outputs = nn.ModuleDict(outputs)
        else:
            raise TypeError('`outputs` is not either list or dict')
        
    def _prepare_in_a(self, data: Dict[str, Tensor]) -> Tensor:
            a_i, a_j = nbops.get_ij(data['a'], data)
            if self.d2features:
                a_j = a_j.transpose(-3, -1).contiguous()
            avf_a = self.conv_a(a_j, data['gs'], data['gv'])
            if self.d2features:
                a_i = a_i.flatten(-2, -1)
            _in = torch.cat([a_i.squeeze(-2), avf_a], dim=-1)
            return _in
    

    def _prepare_in_q(self, data: Dict[str, Tensor]) -> Tensor:
        q_i, q_j = nbops.get_ij(data['charges'], data)
        avf_q = self.conv_q(q_j, data['gs'], data['gv'])
        _in = torch.cat([q_i.squeeze(-2), avf_q], dim=-1)
        return _in
    
    def _update_q(self, data: Dict[str, Tensor], x: Tensor, delta_q: bool = True) -> Dict[str, Tensor]:
        _q, _f, delta_a = x.split([1, 1, x.shape[-1] - 2], dim=-1)
        if delta_q:
            q = data['charges'] + _q
        else:
            q = _q
        f = _f.pow(2)
        q = ops.nse(data['charge'], q, f, data)
        data['charges'] = q
        data['a'] = data['a'] + delta_a.view_as(data['a'])
        return data    


    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data = self.prepare_input(data)

        # initial features
        a: Tensor = self.afv(data['numbers'])
        if self.d2features:
            a = a.unflatten(-1, (self.nfeature, self.nshifts_s))
        data['a'] = a

        # make sure that charge has channel dimension
        while data['charge'].ndim < 2:
            data['charge'] = data['charge'].unsqueeze(-1)

        # AEV
        data = self.aev(data)

        # MP iterations
        _npass = len(self.mlps)
        for ipass, mlp in enumerate(self.mlps):
            if ipass == 0:
                _in = self._prepare_in_a(data)
            else:
                _in = torch.cat([self._prepare_in_a(data), self._prepare_in_q(data)], dim=-1)

            _out = mlp(_in)
            if data['_input_padded'].item():
                _out = nbops.mask_i_(_out, data, mask_value=0.0)

            if ipass == 0:
                data = self._update_q(data, _out, delta_q=False)
            elif ipass < _npass - 1:
                data = self._update_q(data, _out, delta_q=True)
            else:
                data['aim'] = _out

        # squeeze charges
        data['charge'] = data['charge'].squeeze(-1)
        data['charges'] = data['charges'].squeeze(-1)                

        # readout
        for m in self.outputs.children():
            data = m(data)

        return data





    