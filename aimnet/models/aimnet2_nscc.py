import torch
from torch import nn, Tensor
from typing import List, Dict, Union
from aimnet.aev import AEVSV, AEVSV_NB, ConvSV, ConvSV_NB
from aimnet.modules import MLP, Embedding
from aimnet.ops import calc_pad_mask, nqe


class AIMNet2Q(nn.Module):
    def __init__(self, aev: Dict, nfeature: int, d2features: bool, ncomb_v: int, hidden1: List[int], hidden2: List[int],
                 outputs: Union[List[nn.Module], Dict[str, nn.Module]], nblist=False):
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
            conv1 = ConvSV_NB(nchannel=nfeature, d2features=d2features, **conv_param)
            conv_q = ConvSV_NB(nchannel=1, d2features=False, **conv_param)
        else:
            conv1 = ConvSV(nchannel=nfeature, d2features=d2features, **conv_param)
            conv_q = ConvSV(nchannel=1, d2features=False, **conv_param)
        self.add_module('conv1', conv1)
        self.add_module('conv_q', conv_q)
        mlp_param = {'activation_fn': nn.GELU(), 'last_linear': True}
        mlp1 = MLP(n_in=conv1.output_size() + nfeature_tot,
                   n_out=nfeature_tot + 2, hidden=hidden1, **mlp_param)
        mlp2 = MLP(n_in=conv1.output_size() + conv_q.output_size() +
                   nfeature_tot + 1, n_out=2, hidden=hidden2, **mlp_param)
        self.add_module('mlp1', mlp1)
        self.add_module('mlp2', mlp2)

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

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data = self.prepare_data(data)
        data = self.aev(data)
        if 'idx_j' in data:
            a_j = data['a'][data['idx_j']]
        else:
            a_j = data['a']
        avf = self.conv1(a_j, data['gs'], data['gv'])
        a = data['a']
        if self.d2features:
            a = a.flatten(-2, -1)
        _in = torch.cat([a, avf], dim=-1)
        out = self.mlp1(_in)
        if 'pad_mask' in data and data['pad_mask'].numel() > 1:
            out = out.masked_fill(data['pad_mask'].unsqueeze(-1), 0.0)
        q, f, _a = out.split([1, 1, out.shape[-1] - 2], dim=-1)
        q = q.squeeze(-1)
        f = f.squeeze(-1)
        q = nqe(data['charge'], q, f)
        a = a + _a
        data['a'] = a.view_as(data['a'])
        if 'idx_j' in data:
            a_j = data['a'][data['idx_j']]
            q_j = q[data['idx_j']].unsqueeze(-1)
        else:
            a_j = data['a']
            q_j = q.unsqueeze(-1)
        avfa = self.conv1(a_j, data['gs'], data['gv'])
        avfq = self.conv_q(q_j, data['gs'], data['gv'])
        _in = torch.cat([a, q.unsqueeze(-1), avfa, avfq], dim=-1)
        out = self.mlp2(_in)
        if 'pad_mask' in data and data['pad_mask'].numel() > 1:
            out = out.masked_fill(data['pad_mask'].unsqueeze(-1), 0.0)
        _q, f = out.unbind(-1)
        _q = q.squeeze(-1) + _q
        data['charges'] = nqe(data['charge'], q, f)

        for m in self.outputs.children():
            data = m(data)

        return data


class AIMNet2E(nn.Module):
    def __init__(self, aev: Dict, nfeature: int, d2features: bool, ncomb_v: int, aim_size: int, hidden1: List[int], hidden2: List[int],
                 outputs: Union[List[nn.Module], Dict[str, nn.Module]], share_aev=True, nblist=False):
        super().__init__()
        if nblist:
            self.add_module('aev', AEVSV_NB(**aev))
        else:
            self.add_module('aev', AEVSV(**aev))
        self.share_aev = share_aev
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
            conv_a = ConvSV_NB(nchannel=nfeature, d2features=d2features, **conv_param)
            conv_q = ConvSV_NB(nchannel=1, d2features=False, **conv_param)
        else:
            conv_a = ConvSV(nchannel=nfeature, d2features=d2features, **conv_param)
            conv_q = ConvSV(nchannel=1, d2features=False, **conv_param)
        self.add_module('conv_a', conv_a)
        self.add_module('conv_q', conv_q)

        mlp_param = {'n_in': conv_a.output_size() + conv_q.output_size() + nfeature_tot + 1, 'activation_fn': nn.GELU()}
        mlp1 = MLP(n_out=nfeature_tot + 2, hidden=hidden1, last_linear=True, **mlp_param)
        mlp2 = MLP(n_out=aim_size, hidden=hidden2, last_linear=False, **mlp_param)
        self.add_module('mlp1', mlp1)
        self.add_module('mlp2', mlp2)

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
        data['charges'] = data['charges'].to(torch.float)

        if 'pad_mask' not in data or '_natom' not in data:
            data = calc_pad_mask(data)

        a = self.afv(data['numbers'])
        if self.d2features:
            a = a.unflatten(-1, (self.nfeature, self.nshifts_s))
        data['a'] = a
        return data

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data = self.prepare_data(data)
        if not self.share_aev or 'gv' not in data:
            data = self.aev(data)
        a = data['a']
        if self.d2features:
            a = a.flatten(-2, -1)
        q = data['charges']
        if 'idx_j' in data:
            a_j = data['a'][data['idx_j']]
            q_j = q[data['idx_j']].unsqueeze(-1)
        else:
            a_j = data['a']
            q_j = q.unsqueeze(-1)
        avfa = self.conv_a(a_j, data['gs'], data['gv'])
        avfq = self.conv_q(q_j, data['gs'], data['gv'])
        _in = torch.cat([a, q.unsqueeze(-1), avfa, avfq], dim=-1)
        out = self.mlp1(_in)
        if 'pad_mask' in data and data['pad_mask'].numel() > 1:
            out = out.masked_fill(data['pad_mask'].unsqueeze(-1), 0.0)
        _q, f, _a = out.split([1, 1, out.shape[-1] - 2], dim=-1)
        _q = _q.squeeze(-1)
        f = f.squeeze(-1)
        q = q + _q
        q = nqe(data['charge'], q, f)
        data['charges'] = q
        a = a + _a
        data['a'] = a.view_as(data['a'])
        if 'idx_j' in data:
            a_j = data['a'][data['idx_j']]
            q_j = q[data['idx_j']].unsqueeze(-1)
        else:
            a_j = data['a']
            q_j = q.unsqueeze(-1)
        avfa = self.conv_a(a_j, data['gs'], data['gv'])
        avfq = self.conv_q(q_j, data['gs'], data['gv'])
        _in = torch.cat([a, q.unsqueeze(-1), avfa, avfq], dim=-1)
        data['aim'] = self.mlp2(_in)

        for m in self.outputs.children():
            data = m(data)

        return data


class AIMNet2QE(nn.Module):
    def __init__(self, qnet: Dict, enet: Dict, outputs: Union[List[nn.Module], Dict[str, nn.Module]], detached_q=True):
        super().__init__()
        self.add_module('qnet', AIMNet2Q(**qnet))
        self.add_module('enet', AIMNet2E(**enet))

        if isinstance(outputs, list):
            self.outputs = nn.ModuleList(outputs)
        elif isinstance(outputs, dict):
            self.outputs = nn.ModuleDict(outputs)
        else:
            raise TypeError('`outputs` is not either list or dict')

        self.detached_q = detached_q

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        q = []
        data = self.qnet(data)
        q.append(data['charges'])
        if self.detached_q:
            data['charges'] = data['charges'].detach()
        data = self.enet(data)
        q.append(data['charges'])
        data['charges'] = torch.stack(q, dim=0)
        for o in self.outputs.children():
            data = o(data)
        return data


class AIMNet2ENoChg(nn.Module):
    def __init__(self, aev: Dict, nfeature: int, d2features: bool, ncomb_v: int, aim_size: int, hidden1: List[int], hidden2: List[int],
                 outputs: Union[List[nn.Module], Dict[str, nn.Module]], nblist=False):
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
            conv_a = ConvSV_NB(nchannel=nfeature, d2features=d2features, **conv_param)
            conv_q = ConvSV_NB(nchannel=1, d2features=False, **conv_param)
        else:
            conv_a = ConvSV(nchannel=nfeature, d2features=d2features, **conv_param)
            conv_q = ConvSV(nchannel=1, d2features=False, **conv_param)
        self.add_module('conv_a', conv_a)
        self.add_module('conv_q', conv_q)

        mlp_param = {'n_in': conv_a.output_size() + nfeature_tot, 'activation_fn': nn.GELU()}
        mlp1 = MLP(n_out=nfeature_tot + 2, hidden=hidden1, last_linear=True, **mlp_param)
        mlp_param = {'n_in': conv_a.output_size() + conv_q.output_size() + nfeature_tot + 1, 'activation_fn': nn.GELU()}
        mlp2 = MLP(n_out=aim_size, hidden=hidden2, last_linear=False, **mlp_param)
        self.add_module('mlp1', mlp1)
        self.add_module('mlp2', mlp2)

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

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data = self.prepare_data(data)
        data = self.aev(data)
        a = data['a']
        if self.d2features:
            a = a.flatten(-2, -1)
        if 'idx_j' in data:
            a_j = data['a'][data['idx_j']]
        else:
            a_j = data['a']
        avfa = self.conv_a(a_j, data['gs'], data['gv'])
        _in = torch.cat([a, avfa], dim=-1)
        out = self.mlp1(_in)
        if 'pad_mask' in data and data['pad_mask'].numel() > 1:
            out = out.masked_fill(data['pad_mask'].unsqueeze(-1), 0.0)
        q, f, _a = out.split([1, 1, out.shape[-1] - 2], dim=-1)
        q = q.squeeze(-1)
        f = f.squeeze(-1)
        q = nqe(data['charge'], q, f)
        data['charges'] = q
        a = a + _a
        data['a'] = a.view_as(data['a'])
        if 'idx_j' in data:
            a_j = data['a'][data['idx_j']]
            q_j = q[data['idx_j']].unsqueeze(-1)
        else:
            a_j = data['a']
            q_j = q.unsqueeze(-1)
        avfa = self.conv_a(a_j, data['gs'], data['gv'])
        avfq = self.conv_q(q_j, data['gs'], data['gv'])
        _in = torch.cat([a, q.unsqueeze(-1), avfa, avfq], dim=-1)
        data['aim'] = self.mlp2(_in)

        for m in self.outputs.children():
            data = m(data)

        return data
