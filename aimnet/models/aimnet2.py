import torch
from torch import nn, Tensor
from typing import List, Dict, Tuple, Union
from aimnet.aev import AEVSV, AEVSV_NB, ConvSV, ConvSV_NB
from aimnet.modules import MLP, Embedding
from aimnet.ops import calc_pad_mask, nqe


class AIMNet2(nn.Module):
    def __init__(self, aev: Dict, nfeature: int, d2features: bool, ncomb_v: int, hidden: Tuple[List[int]], 
                 aim_size: int, outputs: Union[List[nn.Module], Dict[str, nn.Module]], nblist=False):
        super().__init__()

        if nblist:
            self.add_module('aev', AEVSV_NB(**aev))
        else:
            self.add_module('aev', AEVSV(**aev))
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

        self.add_module('afv', Embedding(num_embeddings=64, embedding_dim=nfeature, padding_idx=0))
        with torch.no_grad():
            nn.init.orthogonal_(self.afv.weight[1:])
            if d2features:
                self.afv.weight = nn.Parameter(self.afv.weight.clone().unsqueeze(-1).expand(64, nfeature, nshifts_s).flatten(-2, -1))

        conv_param = dict(nshifts_s=nshifts_s, nshifts_v=nshifts_v,
                          ncomb_v=ncomb_v, do_vector=True)
        if nblist:
            self.conv_a = ConvSV_NB(nchannel=nfeature, d2features=d2features, **conv_param)
            self.conv_q = ConvSV_NB(nchannel=1, d2features=False, **conv_param)
        else:
            self.conv_a = ConvSV(nchannel=nfeature, d2features=d2features, **conv_param)
            self.conv_q = ConvSV(nchannel=1, d2features=False, **conv_param)

        mlp_param = {'activation_fn': nn.GELU(), 'last_linear': True}
        mlps = [MLP(n_in=self.conv_a.output_size() + nfeature_tot,
                   n_out=nfeature_tot+2, hidden=hidden[0], **mlp_param)]
        mlp_param = {'activation_fn': nn.GELU(), 'last_linear': False}
        for h in hidden[1:-1]:
            mlps.append(MLP(n_in=self.conv_a.output_size() + self.conv_q.output_size() +
                   nfeature_tot+1, n_out=nfeature_tot+2, hidden=h, **mlp_param))
        mlp_param = {'activation_fn': nn.GELU(), 'last_linear': False}
        mlps.append(MLP(n_in=self.conv_a.output_size() + self.conv_q.output_size() +
                   nfeature_tot+1, n_out=aim_size, hidden=hidden[-1], **mlp_param))
        self.mlps = nn.ModuleList(mlps)

        if isinstance(outputs, list):
            self.outputs = nn.ModuleList(outputs)
        elif isinstance(outputs, dict):
            self.outputs = nn.ModuleDict(outputs)
        else:
            raise TypeError('`outputs` is not either list or dict')

    def prepare_data(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data['coord'] = data['coord'].to(torch.float)
        data['numbers'] = data['numbers'].to(torch.long)
        data['charge'] = data['charge'].to(torch.float)

        if 'pad_mask' not in data or '_natom' not in data:
            data = calc_pad_mask(data)

        a = self.afv(data['numbers'])
        if self.d2features:
            a = a.unflatten(-1, (self.nfeature, self.nshifts_s))
        data['a'] = a
        return data

    def _prepare_in_a(self, data: Dict[str, Tensor]) -> Tensor:
        a_i = data['a']
        if 'idx_j' in data:
            # a_j = a_i[data['idx_j']]
            _s0, _s1, _s2, _s3 = data['idx_j'].shape[0], data['idx_j'].shape[1], a_i.shape[-2], a_i.shape[-1]
            a_j = torch.index_select(a_i, 0, data['idx_j'].flatten()).view(_s0, _s1, _s2, _s3)
        else:
            a_j = a_i
        if self.d2features:
            a_i = a_i.flatten(-2, -1)
        avf_a = self.conv_a(a_j, data['gs'], data['gv'])
        _in = torch.cat([a_i, avf_a], dim=-1)
        return _in

    def _prepare_in_q(self, data: Dict[str, Tensor]) -> Tensor:
        q_i = data['charges'].unsqueeze(-1)
        if 'idx_j' in data:
            # q_j = q_i[data['idx_j']]
            _s0, _s1 = data['idx_j'].shape[0], data['idx_j'].shape[1]
            q_j = torch.index_select(q_i, 0, data['idx_j'].flatten()).view(_s0, _s1, 1)
        else:
            q_j = q_i
        avf_q = self.conv_q(q_j, data['gs'], data['gv'])
        _in = torch.cat([q_i, avf_q], dim=-1)
        return _in

    def _zero_padded(self, data: Dict[str, Tensor], x: Tensor) -> Tensor:
        if 'pad_mask' in data and data['pad_mask'].numel() > 1:
            x = x.masked_fill(data['pad_mask'].unsqueeze(-1), 0.0)
        return x

    def _update_q(self, data: Dict[str, Tensor], x: Tensor, delta_q: bool = True) -> Dict[str, Tensor]:
        _q, f, delta_a = x.split([1, 1, x.shape[-1] - 2], dim=-1)
        _q = _q.squeeze(-1)
        f = f.squeeze(-1)
        if delta_q:
            q = data['charges'] + _q
        else:
            q = _q
        q = nqe(data['charge'], q, f)
        data['charges'] = q
        data['a'] = data['a'] + delta_a.view_as(data['a'])
        return data

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data = self.prepare_data(data)
        data = self.aev(data)

        _npass = len(self.mlps)
        for ipass, mlp in enumerate(self.mlps):
            if ipass == 0:
                _in = self._prepare_in_a(data)
            else:
                _in = torch.cat([self._prepare_in_a(data), self._prepare_in_q(data)], dim=-1)

            _out = mlp(_in)
            _out = self._zero_padded(data, _out)

            if ipass == 0:
                data = self._update_q(data, _out, delta_q=False)
            elif ipass < _npass - 1:
                data = self._update_q(data, _out, delta_q=True)
            else:
                data['aim'] = _out

        for m in self.outputs.children():
            data = m(data)

        return data
