from typing import Any, Dict, List, Optional, Union, Callable
import math
from numpy import dtype
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.autograd import grad
from aimnet import ops
from aimnet.config import get_init_module, get_module


def MLP(n_in: int, n_out: int,
        hidden: List[int] = [],
        activation_fn: Union[Callable, str] = 'torch.nn.GELU',
        activation_kwargs: Dict[str, Any] = {},
        weight_init_fn: Union[Callable, str] = 'torch.nn.init.xavier_normal_',
        bias: bool = True, last_linear: bool = True
        ):
    """ Convenienvce function to build MLP from config
    """
    if isinstance(activation_fn, str):
        activation_fn = get_init_module(
            activation_fn, kwargs=activation_kwargs)
    assert callable(activation_fn)
    if isinstance(weight_init_fn, str):
        weight_init_fn = get_module(weight_init_fn)
    assert callable(weight_init_fn)
    sizes = [n_in, *hidden, n_out]
    layers = list()
    for i in range(1, len(sizes)):
        n_in, n_out = sizes[i-1], sizes[i]
        l = nn.Linear(n_in, n_out, bias=bias)
        with torch.no_grad():
            weight_init_fn(l.weight)
            if bias:
                nn.init.zeros_(l.bias)
        layers.append(l)
        if not (last_linear and i == len(sizes) - 1):
            layers.append(activation_fn)
    return nn.Sequential(*layers)


class Embedding(nn.Embedding):
    def __init__(self, init: Optional[Dict[int, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        with torch.no_grad():
            if init is not None:
                for i in range(self.weight.shape[0]):
                    if self.padding_idx is not None and i == self.padding_idx:
                        continue
                    if i in init:
                        self.weight[i] = init[i]
                    else:
                        self.weight[i].fill_(float('nan'))
                for k, v in init.items():
                    self.weight[k] = v

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)


class AtomicShift(nn.Module):
    def __init__(self, key_in: str, key_out: str, num_types: int = 64,
                 dtype: torch.dtype = torch.float, requires_grad: bool = True, reduce_sum=False):
        super().__init__()
        shifts = nn.Embedding(num_types, 1, padding_idx=0)
        with torch.no_grad():
            shifts.weight = shifts.weight.to(dtype)
            nn.init.zeros_(shifts.weight)
        shifts.weight.requires_grad_(requires_grad)
        self.add_module('shifts', shifts)
        self.key_in = key_in
        self.key_out = key_out
        self.reduce_sum = reduce_sum

    def extra_repr(self) -> str:
        return f'key_in: {self.key_in}, key_out: {self.key_out}'

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        shifts = self.shifts(data['numbers'].to(torch.long)).squeeze(-1)
        if self.reduce_sum:
            shifts = shifts.sum(-1)
        data[self.key_out] = data[self.key_in] + shifts
        return data


class AtomicSum(nn.Module):
    def __init__(self, key_in: str, key_out: str):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out

    def extra_repr(self) -> str:
        return f'key_in: {self.key_in}, key_out: {self.key_out}'

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = data[self.key_in].sum(-1, dtype=torch.double)
        data[self.key_out] = x
        return data


class AtomicSumNB(AtomicSum):
    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        src = data[self.key_in]
        idx = data['mol_idx']
        data[self.key_out] = torch.zeros(idx.max() + 1, device=src.device).scatter_add_(0, idx, src)
        return data


class Dipole(nn.Module):
    def __init__(self, key_in: str = 'charges',
                 key_out: str = 'dipole',
                 center_coord: bool = False):
        super().__init__()
        self.center_coord = center_coord
        self.key_out = key_out
        self.key_in = key_in
        mass = [0.,   1.0079,   4.0026,   6.941,   9.0122,  10.811,
                12.0107,  14.0067,  15.9994,  18.9984,  20.1797,  22.9897,
                24.305,  26.9815,  28.0855,  30.9738,  32.065,  35.453,
                39.948,  39.0983,  40.078,  44.9559,  47.867,  50.9415,
                51.9961,  54.938,  55.845,  58.9332,  58.6934,  63.546,
                65.39,  69.723,  72.64,  74.9216,  78.96,  79.904,
                83.8,  85.4678,  87.62,  88.9059,  91.224,  92.9064,
                95.94,  98., 101.07, 102.9055, 106.42, 107.8682,
                112.411, 114.818, 118.71, 121.76, 127.6, 126.9045,
                131.293, 132.9055, 137.327, 138.9055, 140.116, 140.9077,
                144.24, 145., 150.36, 151.964, 157.25, 158.9253,
                162.5, 164.9303, 167.259, 168.9342, 173.04, 174.967,
                178.49, 180.9479, 183.84, 186.207, 190.23, 196.9665,
                192.217, 195.078, 200.59, 204.3833, 207.2, 208.9804,
                209., 210., 222., 223., 226., 227.,
                232.0381, 231.0359, 238.0289, 237., 244., 243.,
                247., 247., 251., 252., 257., 258.,
                259., 262., 261., 262., 266., 264.,
                277., 268., 261.9, 271.8, 285., 286.,
                289., 288., 293., 260.9, 294.]
        self.register_parameter('mass', nn.Parameter(
            torch.tensor(mass, dtype=torch.float), requires_grad=False))

    def extra_repr(self) -> str:
        return f'key_in: {self.key_in}, key_out: {self.key_out}, center_coord: {self.center_coord}'

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        q = data[self.key_in]
        r = data['coord']
        if self.center_coord:
            m = self.mass[data['numbers']].unsqueeze(-1)
            c = (r * m).sum(dim=-2) / m.sum(dim=-2)
            r = r - c.unsqueeze(-2)
        data[self.key_out] = (q.unsqueeze(-1) * r).sum(dim=-2)
        return data
    

class Quadrupole(Dipole):
    def __init__(self, key_in: str = 'charges',
                 key_out: str = 'quadrupole',
                 center_coord: bool = False
                 ):
        super().__init__(key_in=key_in, key_out=key_out, center_coord=center_coord)

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        q = data[self.key_in]
        r = data['coord']
        if self.center_coord:
            m = self.mass[data['numbers']].unsqueeze(-1)
            c = (r * m).sum(dim=-2) / m.sum(dim=-2)
            r = r - c.unsqueeze(-2)
        _x = torch.cat([r.pow(2), r * r.roll(-1, -1)], dim=-1)
        quad = (q.unsqueeze(-1) * _x).sum(dim=-2)
        _x1, _x2 = quad.split(3, dim=-1)
        _x1 = _x1 - _x1.mean(dim=-1, keepdim=True)
        quad = torch.cat([_x1, _x2], dim=-1)
        data[self.key_out] = quad
        return data


class Output(nn.Module):
    def __init__(self, mlp: Union[Dict, nn.Module], n_in: int, n_out: int,
                 key_in: str, key_out: str,
                 apply_fn : str = 'none'):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out
        if not isinstance(mlp, nn.Module):
            mlp = MLP(n_in=n_in, n_out=n_out, **mlp)
        self.add_module('mlp', mlp)
        self.apply_fn = apply_fn

    def extra_repr(self) -> str:
        return f'key_in: {self.key_in}, key_out: {self.key_out}'

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        v = self.mlp(data[self.key_in])
        if 'pad_mask' in data and data['pad_mask'].numel() > 1:
            v = v.masked_fill(data['pad_mask'].unsqueeze(-1), 0.0)
        v = v.squeeze(-1)
        if self.apply_fn == 'pow2':
            v = v.pow(2)
        data[self.key_out] = v
        return data


class NQE(nn.Module):
    def __init__(self, key_in: str, key_out: str):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        q, f = data[self.key_in].unbind(dim=-1)
        q = ops.nqe(data['charge'], q, f)
        data[self.key_out] = q
        return data


class ExpMult(nn.Module):
    def __init__(self, key_in: str, key_out: str, data: Dict[int, float] = None):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out
        self.register_parameter('weights', nn.Parameter(
           torch.zeros(128), requires_grad=False))
        if data is not None:
            for k, v in data.items():
                self.weights[k] = v

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data[self.key_out] = self.weights[data['numbers']] * data[self.key_in].exp()
        return data


class Forces(nn.Module):
    def __init__(self, module: nn.Module,
                 x: str = 'coord', y: str = 'energy', key_out: str = 'forces', ipass: int = -1,
                 multipass_module=False):
        super().__init__()
        self.add_module('module', module)
        self.x = x
        self.y = y
        self.key_out = key_out
        self.ipass = ipass
        self.multipass_module = multipass_module

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        data[self.x].requires_grad_(True)
        data = self.module(data)
        if self.multipass_module:
            y = data[self.y][self.ipass]
        else:
            y = data[self.y]
        g = torch.autograd.grad(
            [y.sum()], [data[self.x]], create_graph=self.training)[0]
        assert g is not None
        data[self.key_out] = - g
        torch.set_grad_enabled(prev)
        return data


class SRRep(nn.Module):
    def __init__(self, key_out='e_rep', cutoff_fn='none', rc=5.2, reduce_sum=True):
        super().__init__()
        self.key_out = key_out
        self.cutoff_fn = cutoff_fn
        self.reduce_sum = reduce_sum

        self.register_parameter('rc', nn.Parameter(
            torch.tensor(rc), requires_grad=False))

        # GFN1 parameters
        # alpha
        repa = torch.tensor([
            0.000001, 2.209700, 1.382907, 0.671797, 0.865377, 1.093544,
            1.281954, 1.727773, 2.004253, 2.507078, 3.038727, 0.704472,
            0.862629, 0.929219, 0.948165, 1.067197, 1.200803, 1.404155,
            1.323756, 0.581529, 0.665588, 0.841357, 0.828638, 1.061627,
            0.997051, 1.019783, 1.137174, 1.188538, 1.399197, 1.199230,
            1.145056, 1.047536, 1.129480, 1.233641, 1.270088, 1.153580,
            1.335287, 0.554032, 0.657904, 0.760144, 0.739520, 0.895357,
            0.944064, 1.028240, 1.066144, 1.131380, 1.206869, 1.058886,
            1.026434, 0.898148, 1.008192, 0.982673, 0.973410, 0.949181,
            1.074785, 0.579919, 0.606485, 1.311200, 0.839861, 0.847281,
            0.854701, 0.862121, 0.869541, 0.876961, 0.884381, 0.891801,
            0.899221, 0.906641, 0.914061, 0.921481, 0.928901, 0.936321,
            0.853744, 0.971873, 0.992643, 1.132106, 1.118216, 1.245003,
            1.304590, 1.293034, 1.181865, 0.976397, 0.988859, 1.047194,
            1.013118, 0.964652, 0.998641
        ])
        # Zeff
        repb = torch.tensor([
            0.000000,  1.116244,  0.440231,  2.747587,  4.076830,  4.458376,
            4.428763,  5.498808,  5.171786,  6.931741,  9.102523, 10.591259,
            15.238107, 16.283595, 16.898359, 15.249559, 15.100323, 17.000000,
            17.153132, 20.831436, 19.840212, 18.676202, 17.084130, 22.352532,
            22.873486, 24.160655, 25.983149, 27.169215, 23.396999, 29.000000,
            31.185765, 33.128619, 35.493164, 36.125762, 32.148852, 35.000000,
            36.000000, 39.653032, 38.924904, 39.000000, 36.521516, 40.803132,
            41.939347, 43.000000, 44.492732, 45.241537, 42.105527, 43.201446,
            49.016827, 51.718417, 54.503455, 50.757213, 49.215262, 53.000000,
            52.500985, 65.029838, 46.532974, 48.337542, 30.638143, 34.130718,
            37.623294, 41.115870, 44.608445, 48.101021, 51.593596, 55.086172,
            58.578748, 62.071323, 65.563899, 69.056474, 72.549050, 76.041625,
            55.222897, 63.743065, 74.000000, 75.000000, 76.000000, 77.000000,
            78.000000, 79.000000, 80.000000, 81.000000, 79.578302, 83.000000,
            84.000000, 85.000000, 86.000000
        ])
        _angs_to_au = 1.8897161646320724
        _au_to_ev = 27.211396641308
        # convert params to Angs anf eV
        # pre-compute sqrt(alpha)
        # pre-multiply by 0.5
        repa = repa.pow(0.5) * _angs_to_au ** 0.75
        repb = repb * (0.5 * _au_to_ev / _angs_to_au) ** 0.5
        weight = torch.stack([repa, repb], axis=-1)
        self.params = nn.Embedding(87, 2, padding_idx=0, _weight=weight)
        self.params.weight.requires_grad_(False)

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        p = self.params(data['numbers'].to(torch.long))
        p_ij = p.unsqueeze(-2) * p.unsqueeze(-3)
        alpha_ij, zeff_ij = p_ij.unbind(-1)
        e = torch.exp(- alpha_ij *
                      data['d_ij'].pow(1.5)) * zeff_ij / data['d_ij']
        if self.cutoff_fn == 'exp_cutoff':
            e = e * ops.exp_cutoff(data['d_ij'], self.rc)
        elif self.cutoff_fn == 'cosine_cutoff':
            e = e * ops.cosine_cutoff(data['d_ij'], self.rc)
        if self.reduce_sum:
            e = e.flatten(-2, -1)
        e = e.sum(-1, dtype=torch.double)
        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e
        else:
            data[self.key_out] = e
        return data 


class SRRep_NB(SRRep):
    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        p_i = self.params(data['numbers'])
        _s0, _s1, = data['idx_j'].shape
        p_j = torch.index_select(p_i, 0, data['idx_j'].flatten()).view(_s0, _s1, 2)
        # p_j = p_i[data['idx_j']]
        p_ij = p_i.unsqueeze(-2) * p_j
        alpha_ij, zeff_ij = p_ij.unbind(-1)
        e = torch.exp(- alpha_ij * data['d_ij'].pow(1.5)) * zeff_ij / data['d_ij']
        if self.cutoff_fn == 'exp_cutoff':
            e = e * ops.exp_cutoff(data['d_ij'], self.rc)
        else:
            e = e * ops.cosine_cutoff(data['d_ij'], self.rc)
        if self.reduce_sum:
            e = e.flatten(-2, -1)
        e = e.sum(-1, dtype=torch.double)
        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e
        else:
            data[self.key_out] = e
        return data     


class Coulomb(nn.Module):
    def __init__(self, key_in: str = 'charges', key_out: str = 'e_h'):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        d_ij = data['d_ij'] #.clamp(min=0.3)
        q = data[self.key_in]
        q_ij = q.unsqueeze(-1) * q.unsqueeze(-2)
        # ghost atom charges should be already zero!
        # q_ij = q_ij.masked_fill(torch.eye(q_ij.shape[-1], dtype=torch.bool, device=q_ij.device), 0.0)
        e_h = (7.1998226 * q_ij / d_ij).flatten(-2, -1).sum(-1, dtype=torch.double)
        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e_h
        else:
            data[self.key_out] = e_h
        return data


class LRCoulomb(nn.Module):
    def __init__(self, key_in: str = 'charges', key_out: str = 'e_h', rc: float = 4.6):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out
        self.register_parameter('rc', nn.Parameter(torch.tensor(rc), requires_grad=False))

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        d_ij = data['d_ij']
        q = data[self.key_in]
        q_ij = q.unsqueeze(-1) * q.unsqueeze(-2)
        fc = 1.0 - ops.exp_cutoff(d_ij, self.rc)
        q_ij = q_ij.masked_fill(torch.eye(q_ij.shape[-1], dtype=torch.bool, device=q_ij.device), 0.0)
        e_h = (7.1998226 * fc * q_ij / d_ij).flatten(-2, -1).sum(-1, dtype=torch.double)
        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e_h
        else:
            data[self.key_out] = e_h
        return data


class LRCoulomb_NB(LRCoulomb):
    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if 'd_ij_coul' not in data:
            coord_i = data['coord']
            _s0, _s1 = data['idx_j_coul'].shape
            coord_j = torch.index_select(coord_i, 0, data['idx_j_coul'].flatten()).view(_s0, _s1, 3)
            # coord_j = coord_i[data['idx_j_coul']]
            if 'shifts_coul' in data:
                shifts = data['shifts_coul'] @ data['cell']
                coord_j = coord_j + shifts
            r_ij = coord_j - coord_i.unsqueeze(-2)
            d_ij2 = r_ij.pow(2).sum(-1)
            d_ij2 = d_ij2.masked_fill(data['nb_pad_mask_coul'], 1.0)
            data['d_ij_coul'] = d_ij2.sqrt()
        d_ij = data['d_ij_coul']

        q_i = data[self.key_in]
        _s0, _s1 = data['idx_j_coul'].shape
        q_j = torch.index_select(q_i, 0, data['idx_j_coul'].flatten()).view(_s0, _s1)
        #q_j = q_i[data['idx_j_coul']]
        q_j = q_j.masked_fill(data['nb_pad_mask_coul'], 0.0)
        q_ij = q_i.unsqueeze(-1) * q_j
        fc = 1 - ops.exp_cutoff(d_ij, self.rc)
        eh = (7.1998226 * fc * q_ij / d_ij).flatten(-2, -1).sum(-1, dtype=torch.double)
        if self.key_out in data:
            data[self.key_out] = data[self.key_out] - eh
        else:
            data[self.key_out] = eh
        return data    


class NegLRCoulomb_NB(LRCoulomb):
    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
       d_ij = data['d_ij']
       q_i = data[self.key_in]
       _s0, _s1 = data['idx_j'].shape
       q_j = torch.index_select(q_i, 0, data['idx_j'].flatten()).view(_s0, _s1)
       #q_j = q_i[data['idx_j']]
       q_j = q_j.masked_fill(data['nb_pad_mask'], 0.0)
       q_ij = q_i.unsqueeze(-1) * q_j
       fc = ops.exp_cutoff(d_ij, self.rc)
       eh = (7.1998226 * fc * q_ij / d_ij).flatten(-2, -1).sum(-1, dtype=torch.double)
       if self.key_out in data:
           data[self.key_out] = data[self.key_out] - eh
       else:
           data[self.key_out] = eh
       return data


class DispParam(nn.Module):
    def __init__(self, ref='d4'):
        super().__init__()
        disp_param0 = torch.zeros(64, 2)
        _c6_d4 = [1.5122887314917994, 41.40182387769873, 27.252285365389284, 28.010607461450974, 20.756414983658626,
                11.236458665855995, 167.93099777171867, 168.0238807938331, 139.3850326536157, 99.7981480487368,
                236.68075142406505, 227.19301011811083, 182.5765835027261, 372.6816957649393]
        _c6_d3 = [3.1048481, 33.77778, 21.813042, 17.24206, 11.789205, 7.220059,
                 159.96632, 155.46225, 126.68423, 90.484695, 227.679, 211.05339, 169.0966, 358.20947]
        if ref == 'd4':
            _c6 = _c6_d4
        elif ref == 'd3':
            _c6 = _c6_d3
        else:
            raise ValueError(f'Unknown c6 reference: {ref}')
        for i, c, a in zip(
                [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53],
                _c6,
                [1.9110786, 11.809439, 8.214585, 7.743854, 6.041415, 3.8817358, 28.134447, 23.829082,
                 19.99253, 15.286761, 28.327911, 26.247684, 21.840067, 33.60986]
                ):
            disp_param0[i, 0] = c
            disp_param0[i, 1] = a
        disp_param0[0, 1] = 1.0
        self.register_parameter('disp_param0', nn.Parameter(disp_param0, requires_grad=False))

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data['disp_param'] = data['disp_param'].clamp(min=-4, max=4)
        disp_param_mult = data['disp_param'].exp()
        disp_param = self.disp_param0[data['numbers']]
        c6i, alpha = (disp_param * disp_param_mult).unbind(-1)
        if data['pad_mask'].numel() > 1:
            alpha = alpha.masked_fill(data['pad_mask'], 0.0)
            c6i = c6i.masked_fill(data['pad_mask'], 0.0)
        data['c6i'] = c6i
        data['alpha'] = alpha
        return data


class D3BJ(nn.Module):
    def __init__(self, a1: float, a2: float, s8: float, s6: float = 1.0, key_in_c6='dftd3_c6', key_in_alpha='dftd4_alpha', key_out='disp_energy'):
        super().__init__()

        ## https://github.com/dftd4/dftd4/blob/main/src/dftd4/data/r4r2.f90
        sqrt_z_r4_over_r2 = [0.0,
            8.0589 , 3.4698 ,
            29.0974 ,14.8517 ,11.8799 , 7.8715 , 5.5588 , 4.7566 , 3.8025 , 3.1036 ,
            26.1552 ,17.2304 ,17.7210 ,12.7442 , 9.5361 , 8.1652 , 6.7463 , 5.6004 ,
            29.2012 ,22.3934 ,
            19.0598 ,16.8590 ,15.4023 ,12.5589 ,13.4788 ,
            12.2309 ,11.2809 ,10.5569 ,10.1428 , 9.4907 ,
            13.4606 ,10.8544 , 8.9386 , 8.1350 , 7.1251 , 6.1971 ,
            30.0162 ,24.4103 ,
            20.3537 ,17.4780 ,13.5528 ,11.8451 ,11.0355 ,
            10.1997 , 9.5414 , 9.0061 , 8.6417 , 8.9975 ,
            14.0834 ,11.8333 ,10.0179 , 9.3844 , 8.4110 , 7.5152 ,
            32.7622 ,27.5708 ,
            23.1671 ,21.6003 ,20.9615 ,20.4562 ,20.1010 ,19.7475 ,19.4828 ,
            15.6013 ,19.2362 ,17.4717 ,17.8321 ,17.4237 ,17.1954 ,17.1631 ,
            14.5716 ,15.8758 ,13.8989 ,12.4834 ,11.4421 ,
            10.2671 , 8.3549 , 7.8496 , 7.3278 , 7.4820 ,
            13.5124 ,11.6554 ,10.0959 , 9.7340 , 8.8584 , 8.0125 ,
            29.8135 ,26.3157 ,
            19.1885 ,15.8542 ,16.1305 ,15.6161 ,15.1226 ,16.1576 , 0.0000 ,
            0.0000 , 0.0000 , 0.0000 , 0.0000 , 0.0000 , 0.0000 , 0.0000 ,
            0.0000 , 0.0000 , 0.0000 , 0.0000 , 0.0000 ,
            0.0000 , 0.0000 , 0.0000 , 0.0000 , 5.4929 ,
            6.7286 , 6.5144 ,10.9169 ,10.3600 , 9.4723 , 8.6641  ]

        r4r2 = (0.5 * torch.tensor(sqrt_z_r4_over_r2) * torch.arange(len(sqrt_z_r4_over_r2)).sqrt()).sqrt()
        self.register_parameter('r4r2', nn.Parameter(r4r2, requires_grad=False))

        self.a1 = a1
        self.a2 = a2
        self.s6 = s6
        self.s8 = s8
        self.key_in_c6 = key_in_c6
        self.key_in_alpha = key_in_alpha
        self.key_out = key_out

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        d_ij = data['d_ij'] * 1.889716
        c6i = data[self.key_in_c6]
        alpha = data[self.key_in_alpha]
        numbers = data['numbers']
        
        c6_i, c6_j = c6i.unsqueeze(-1), c6i.unsqueeze(-2)
        alpha_i, alpha_j = alpha.unsqueeze(-1), alpha.unsqueeze(-2)
        c6ij = 2 * c6_i * c6_j / ((c6_i * alpha_j / (alpha_i + 1e-6) + c6_j * alpha_i / (alpha_j + 1e-6)) + 1e-6)

        if 'pad_mask' in data:
            mask = data['pad_mask']
        else:
            mask = None
        if mask is not None and mask.numel() > 1:
            mask = (mask.unsqueeze(-2) + mask.unsqueeze(-1)) | \
                torch.eye(c6ij.shape[1], device=d_ij.device,
                        dtype=torch.bool).unsqueeze(0)
        else:
            mask = torch.eye(
                c6ij.shape[1], device=c6ij.device, dtype=torch.bool).unsqueeze(0)
        c6ij = c6ij.masked_fill(mask, 0.0)

        rrii = self.r4r2[numbers] 
        rrij = 3 * rrii.unsqueeze(-2) * rrii.unsqueeze(-1)
        r0ij = self.a1 * rrij.sqrt() + self.a2

        e_ij = c6ij * (self.s6 / (d_ij.pow(6) + r0ij.pow(6)) + self.s8 * rrij / (d_ij.pow(8) + r0ij.pow(8)))
        e_disp = - (0.5 * 27.211368) *  e_ij.flatten(-2, -1).sum(-1, dtype=torch.double) # in eV

        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e_disp
        else:
            data[self.key_out] = e_disp

        return data


class D3BJ_NB(D3BJ):

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if 'd_ij_coul' not in data:
            coord_i = data['coord']
            _s0, _s1 = data['idx_j_coul'].shape
            coord_j = torch.index_select(coord_i, 0, data['idx_j_coul'].flatten()).view(_s0, _s1, 3)
            #coord_j = coord_i[data['idx_j_coul']]
            if 'shifts_coul' in data:
                shifts = data['shifts_coul'] @ data['cell']
                coord_j = coord_j + shifts
            r_ij = coord_j - coord_i.unsqueeze(-2)
            d_ij2 = r_ij.pow(2).sum(-1)
            d_ij2 = d_ij2.masked_fill(data['nb_pad_mask_coul'], 1.0)
            data['d_ij_coul'] = d_ij2.sqrt()
        d_ij = data['d_ij_coul'] * 1.889716
        numbers = data['numbers']

        c6_i = data[self.key_in_c6].unsqueeze(-1)
        _s0, _s1 = data['idx_j_coul'].shape
        c6_j = torch.index_select(data[self.key_in_c6], 0, data['idx_j_coul'].flatten()).view(_s0, _s1)
        #c6_j = data[self.key_in_c6][data['idx_j_coul']]
        alpha_i = data[self.key_in_alpha].unsqueeze(-1).clamp(min=1e-6)
        alpha_j = torch.index_select(data[self.key_in_alpha], 0, data['idx_j_coul'].flatten()).view(_s0, _s1).clamp(min=1e-6)
        #alpha_j = data[self.key_in_alpha][data['idx_j_coul']].clamp(min=1e-6)

        c6ij = 2 * c6_i * c6_j / (c6_i * alpha_j / alpha_i + c6_j * alpha_i / alpha_j).clamp(min=1e-6)

        rrii = self.r4r2[numbers]
        rrii_j = torch.index_select(rrii, 0, data['idx_j_coul'].flatten()).view(_s0, _s1)
        rrij = 3 * rrii.unsqueeze(-1) * rrii_j
        #rrij = 3 * rrii.unsqueeze(-1) * rrii[data['idx_j_coul']]
        r0ij = self.a1 * rrij.sqrt() + self.a2
        d_ij = d_ij.to(torch.double)
        e_ij = c6ij * (self.s6 / (d_ij.pow(6) + r0ij.pow(6)) + self.s8 * rrij / (d_ij.pow(8) + r0ij.pow(8)))
        e_disp = - (0.5 * 27.211368) *  e_ij.flatten(-2, -1).sum(-1, dtype=torch.double) # in eV

        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e_disp
        else:
            data[self.key_out] = e_disp

        return data
    

class CoulombDSFSwitched(nn.Module):
    """
    TensorMol https://doi.org/10.1063/1.4973380
    """
    def __init__(self, key_in: str = 'charges', key_out: str = 'e_h', alpha=0.2, r_sw=4.6, r_cut=15.0):
        super().__init__()
        self.register_parameter('alpha', nn.Parameter(torch.tensor(alpha), requires_grad=False))
        self.register_parameter('r_sw', nn.Parameter(torch.tensor(r_sw), requires_grad=False))
        self.register_parameter('r_cut', nn.Parameter(torch.tensor(r_cut), requires_grad=False))
        self.register_parameter('elu_shift', nn.Parameter(self.dsf_potential(self.r_sw), requires_grad=False))
        self.register_parameter('elu_alpha', nn.Parameter(self.dsf_force(self.r_sw), requires_grad=False))
        self.key_in = key_in
        self.key_out = key_out

    def dsf_potential(self, d_ij):
        Rc = self.r_cut
        alph = self.alpha
        _c1 = (alph * d_ij).erfc() / d_ij
        _c2 = (alph * Rc).erfc() / Rc
        _c3 = _c2 / Rc
        _c4 = 2 * alph * (- (alph * Rc) ** 2).exp() / (Rc * math.pi ** 0.5)
        dsf_pot = (_c1 - _c2 + (d_ij - Rc) * (_c3 + _c4))
        dsf_pot[d_ij > Rc] = 0.0
        return dsf_pot 

    def dsf_force(self, d_ij):
        Rc = self.r_cut
        alph = self.alpha
        dsf_f = (alph * d_ij).erfc() / d_ij.pow(2) \
                - (alph * Rc).erfc() / Rc.pow(2) \
                + 2.0 * alph / (math.pi ** 0.5) * ((-(alph * d_ij).pow(2)).exp() / d_ij - (-(alph * Rc).pow(2)).exp() / (math.pi ** 0.5))
        dsf_f[d_ij > Rc] = 0.0
        return - dsf_f 

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        d_ij = data['d_ij']
        q = data[self.key_in]
        q_ij = q.unsqueeze(-1) * q.unsqueeze(-2)
        q_ij = q_ij.masked_fill(torch.eye(q_ij.shape[-1], dtype=torch.bool, device=q_ij.device), 0.0)
        dsf_pot = self.dsf_potential(d_ij)
        elu_pot = self.elu_alpha * (torch.exp(d_ij - self.r_sw) - 1.0) + self.elu_shift
        pot = torch.where(d_ij < self.r_sw, elu_pot, dsf_pot)
        e_h = 7.1998226 * (q_ij * pot).flatten(-2, -1).sum(-1, dtype=torch.float64)
        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e_h
        else:
            data[self.key_out] = e_h
        return data


class CoulombDSFSwitched_NB(CoulombDSFSwitched):
    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if 'd_ij_coul' not in data:
            coord_i = data['coord']
            _s0, _s1 = data['idx_j_coul'].shape
            coord_j = torch.index_select(coord_i, 0, data['idx_j_coul'].flatten()).view(_s0, _s1, 3)            
            # coord_j = coord_i[data['idx_j_coul']]
            if 'shifts_coul' in data:
                shifts = data['shifts_coul'] @ data['cell']
                coord_j = coord_j + shifts
            r_ij = coord_j - coord_i.unsqueeze(-2)
            d_ij2 = r_ij.pow(2).sum(-1)
            d_ij2 = d_ij2.masked_fill(data['nb_pad_mask_coul'], 1.0)
            data['d_ij_coul'] = d_ij2.sqrt()

        d_ij = data['d_ij_coul']

        q_i = data[self.key_in]
        _s0, _s1 = data['idx_j_coul'].shape
        q_j = torch.index_select(q_i, 0, data['idx_j_coul'].flatten()).view(_s0, _s1)        
        #q_j = q_i[data['idx_j_coul']]
        q_j = q_j.masked_fill(data['nb_pad_mask_coul'] | (d_ij > self.r_cut), 0.0)
        q_ij = q_i.unsqueeze(-1) * q_j

        dsf_pot = self.dsf_potential(d_ij)
        elu_pot = self.elu_alpha * (torch.exp(d_ij - self.r_sw) - 1.0) + self.elu_shift
        pot = torch.where(d_ij < self.r_sw, elu_pot, dsf_pot)
        e_h = 7.1998226 * (q_ij * pot).flatten(-2, -1).sum(-1, dtype=torch.float64)
        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e_h
        else:
            data[self.key_out] = e_h
        return data


class CoulombDSF_NB(nn.Module):
    def __init__(self, key_in: str = 'charges', key_out: str = 'e_h', alpha=0.2):
        super().__init__()
        self.register_parameter('alpha', nn.Parameter(torch.tensor(alpha), requires_grad=False))
        self.key_in = key_in
        self.key_out = key_out

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if 'd_ij_coul' not in data:
            coord_i = data['coord']
            _s0, _s1 = data['idx_j_coul'].shape
            coord_j = torch.index_select(coord_i, 0, data['idx_j_coul'].flatten()).view(_s0, _s1, 3)            
            # coord_j = coord_i[data['idx_j_coul']]
            if 'shifts_coul' in data:
                shifts = data['shifts_coul'] @ data['cell']
                coord_j = coord_j + shifts
            r_ij = coord_j - coord_i.unsqueeze(-2)
            d_ij2 = r_ij.pow(2).sum(-1)
            d_ij2 = d_ij2.masked_fill(data['nb_pad_mask_coul'], 1.0)
            data['d_ij_coul'] = d_ij2.sqrt()

        d_ij = data['d_ij_coul']

        q_i = data[self.key_in]
        _s0, _s1 = data['idx_j_coul'].shape
        q_j = torch.index_select(q_i, 0, data['idx_j_coul'].flatten()).view(_s0, _s1)        
        #q_j = q_i[data['idx_j_coul']]
        q_j = q_j.masked_fill(data['nb_pad_mask_coul'] | (d_ij > data['coul_cutoff']), 0.0)
        q_ij = q_i.unsqueeze(-1) * q_j

        # DSF
        Rc = data['coul_cutoff']
        alph = self.alpha
        _c1 = (alph * d_ij).erfc() / d_ij
        _c2 = (alph * Rc).erfc() / Rc
        _c3 = _c2 / Rc
        _c4 = 2 * alph * (- (alph * Rc) ** 2).exp() / (Rc * math.pi ** 0.5)
        eh_ij = q_ij * (_c1 - _c2 + (d_ij - Rc) * (_c3 + _c4))
        eh = 7.1998226 * eh_ij.flatten(-2, -1).sum(-1, dtype=torch.double)

        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + eh
        else:
            data[self.key_out] = eh

        return data


class CoulombLR_DSF_NB(nn.Module):
    def __init__(self, key_in: str = 'charges', key_out: str = 'e_h', alpha: float = 0.2, rc: float = 4.6):
        super().__init__()
        self.register_parameter('alpha', nn.Parameter(torch.tensor(alpha), requires_grad=False))
        self.register_parameter('rc', nn.Parameter(torch.tensor(rc), requires_grad=False))
        self.key_in = key_in
        self.key_out = key_out

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # short range part
        if 'd_ij' not in data:
            coord_i = data['coord']
            _s0, _s1 = data['idx_j'].shape
            coord_j = torch.index_select(coord_i, 0, data['idx_j'].flatten()).view(_s0, _s1, 3)            
            # coord_j = coord_i[data['idx_j']]
            if 'shifts' in data:
                shifts = data['shifts'] @ data['cell']
                coord_j = coord_j + shifts
            r_ij = coord_j - coord_i.unsqueeze(-2)
            d_ij2 = r_ij.pow(2).sum(-1)
            d_ij2 = d_ij2.masked_fill(data['nb_pad_mask_coul'], 1.0)
            data['d_ij'] = d_ij2.sqrt()
        d_ij = data['d_ij']
        q_i = data[self.key_in]
        _s0, _s1 = data['idx_j'].shape
        q_j = torch.index_select(q_i, 0, data['idx_j'].flatten()).view(_s0, _s1)        
        #q_j = q_i[data['idx_j']]
        q_j = q_j.masked_fill(data['nb_pad_mask'], 0.0)
        q_ij = q_i.unsqueeze(-1) * q_j
        fc = ops.exp_cutoff(d_ij, self.rc)
        eh_short = (7.1998226 * fc * q_ij / d_ij).flatten(-2, -1).sum(-1, dtype=torch.double)

        # DSF part
        if 'd_ij_coul' not in data:
            coord_i = data['coord']
            _s0, _s1 = data['idx_j_coul'].shape
            coord_j = torch.index_select(coord_i, 0, data['idx_j_coul'].flatten()).view(_s0, _s1, 3)            
            # coord_j = coord_i[data['idx_j_coul']]
            if 'shifts_coul' in data:
                shifts = data['shifts_coul'] @ data['cell']
                coord_j = coord_j + shifts
            r_ij = coord_j - coord_i.unsqueeze(-2)
            d_ij2 = r_ij.pow(2).sum(-1)
            d_ij2 = d_ij2.masked_fill(data['nb_pad_mask_coul'], 1.0)
            data['d_ij_coul'] = d_ij2.sqrt()

        d_ij = data['d_ij_coul']
        q_i = data[self.key_in]
        _s0, _s1 = data['idx_j_coul'].shape
        q_j = torch.index_select(q_i, 0, data['idx_j_coul'].flatten()).view(_s0, _s1)        
        #q_j = q_i[data['idx_j_coul']]
        q_j = q_j.masked_fill(data['nb_pad_mask_coul'] | (d_ij > data['coul_cutoff']), 0.0)
        q_ij = q_i.unsqueeze(-1) * q_j

        Rc = data['coul_cutoff']
        alph = self.alpha
        _c1 = (alph * d_ij).erfc() / d_ij
        _c2 = (alph * Rc).erfc() / Rc
        _c3 = _c2 / Rc
        _c4 = 2 * alph * (- (alph * Rc) ** 2).exp() / (Rc * math.pi ** 0.5)
        eh_ij = q_ij * (_c1 - _c2 + (d_ij - Rc) * (_c3 + _c4))
        eh_dsf = 7.1998226 * eh_ij.flatten(-2, -1).sum(-1, dtype=torch.double)

        eh = eh_dsf - eh_short

        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + eh
        else:
            data[self.key_out] = eh

        return data


class CoulombLR_SF_NB(LRCoulomb):
    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # short range part
        if 'd_ij' not in data:
            coord_i = data['coord']
            _s0, _s1 = data['idx_j'].shape
            coord_j = torch.index_select(coord_i, 0, data['idx_j'].flatten()).view(_s0, _s1, 3)            
            # coord_j = coord_i[data['idx_j']]
            if 'shifts' in data:
                shifts = data['shifts'] @ data['cell']
                coord_j = coord_j + shifts
            r_ij = coord_j - coord_i.unsqueeze(-2)
            d_ij2 = r_ij.pow(2).sum(-1)
            d_ij2 = d_ij2.masked_fill(data['nb_pad_mask'], 1.0)
            data['d_ij'] = d_ij2.sqrt()
        d_ij = data['d_ij']
        q_i = data[self.key_in]
        _s0, _s1 = data['idx_j'].shape
        q_j = torch.index_select(q_i, 0, data['idx_j'].flatten()).view(_s0, _s1)        
        #q_j = q_i[data['idx_j']]
        q_j = q_j.masked_fill(data['nb_pad_mask'], 0.0)
        q_ij = q_i.unsqueeze(-1) * q_j
        fc = ops.exp_cutoff(d_ij, self.rc)
        eh_short = (7.1998226 * fc * q_ij / d_ij).flatten(-2, -1).sum(-1, dtype=torch.double)

        # SF part
        if 'd_ij_coul' not in data:
            coord_i = data['coord']
            _s0, _s1 = data['idx_j_coul'].shape
            coord_j = torch.index_select(coord_i, 0, data['idx_j_coul'].flatten()).view(_s0, _s1, 3)            
            # coord_j = coord_i[data['idx_j_coul']]
            if 'shifts_coul' in data:
                shifts = data['shifts_coul'] @ data['cell']
                coord_j = coord_j + shifts
            r_ij = coord_j - coord_i.unsqueeze(-2)
            d_ij2 = r_ij.pow(2).sum(-1)
            d_ij2 = d_ij2.masked_fill(data['nb_pad_mask_coul'], 1.0)
            data['d_ij_coul'] = d_ij2.sqrt()

        d_ij = data['d_ij_coul']
        q_i = data[self.key_in]
        _s0, _s1 = data['idx_j_coul'].shape
        q_j = torch.index_select(q_i, 0, data['idx_j_coul'].flatten()).view(_s0, _s1)        
        #q_j = q_i[data['idx_j_coul']]
        q_j = q_j.masked_fill(data['nb_pad_mask_coul'] | (d_ij > data['coul_cutoff']), 0.0)
        q_ij = q_i.unsqueeze(-1) * q_j

        Rc = data['coul_cutoff']
        _c1 = 1.0 / d_ij
        _c2 = 1.0 / Rc
        _c3 = _c2 / Rc
        eh_ij = q_ij * (1/d_ij - 1/Rc + (d_ij - Rc) * Rc**2)
        eh_sf = 7.1998226 * eh_ij.flatten(-2, -1).sum(-1, dtype=torch.double)

        eh = eh_sf - eh_short

        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + eh
        else:
            data[self.key_out] = eh

        return data
    

class Coulomb_SF_NB(nn.Module):
    def __init__(self, key_in: str = 'charges', key_out: str = 'e_h'):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if 'd_ij_coul' not in data:
            coord_i = data['coord']
            _s0, _s1 = data['idx_j_coul'].shape
            coord_j = torch.index_select(coord_i, 0, data['idx_j_coul'].flatten()).view(_s0, _s1, 3)            
            # coord_j = coord_i[data['idx_j_coul']]
            if 'shifts_coul' in data:
                shifts = data['shifts_coul'] @ data['cell']
                coord_j = coord_j + shifts
            r_ij = coord_j - coord_i.unsqueeze(-2)
            d_ij2 = r_ij.pow(2).sum(-1)
            d_ij2 = d_ij2.masked_fill(data['nb_pad_mask_coul'], 1.0)
            data['d_ij_coul'] = d_ij2.sqrt()

        d_ij = data['d_ij_coul']
        q_i = data[self.key_in]
        _s0, _s1 = data['idx_j_coul'].shape
        q_j = torch.index_select(q_i, 0, data['idx_j_coul'].flatten()).view(_s0, _s1)        
        #q_j = q_i[data['idx_j_coul']]
        q_j = q_j.masked_fill(data['nb_pad_mask_coul'] | (d_ij > data['coul_cutoff']), 0.0)
        q_ij = q_i.unsqueeze(-1) * q_j

        Rc = data['coul_cutoff']

        _c1 = 1.0 / d_ij
        _c2 = 1.0 / Rc
        _c3 = _c2 / Rc
        eh_ij = q_ij * (_c1 - _c2 + (d_ij - Rc) * _c3)
        eh_sf = 7.1998226 * eh_ij.flatten(-2, -1).sum(-1, dtype=torch.double)

        eh = eh_sf 

        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + eh
        else:
            data[self.key_out] = eh

        return data
