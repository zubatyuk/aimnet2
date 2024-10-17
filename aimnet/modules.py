from typing import Any, Dict, List, Tuple, Optional, Union, Callable
import torch
from torch import Tensor, nn
from aimnet import ops, nbops, constants
from aimnet.config import get_init_module, get_module
import math
import os


def MLP(n_in: int, n_out: int,
        hidden: List[int] = [],
        activation_fn: Union[Callable, str] = 'torch.nn.GELU',
        activation_kwargs: Dict[str, Any] = {},
        weight_init_fn: Union[Callable, str] = 'torch.nn.init.xavier_normal_',
        bias: bool = True, last_linear: bool = True
        ):
    """ Convenience function to build MLP from config
    """
    # hp search hack
    hidden = [x for x in hidden if x > 0]
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


class CosineCutoff(nn.Module):
    def __init__(self, rc: float = 5.0):
        super().__init__()
        self.register_buffer('rc', torch.tensor(rc))

    def forward(self, x: Tensor, inverse: bool = False) -> Tensor:
        x = ops.cosine_cutoff(x, self.rc)
        if inverse:
            x = 1 - x
        return x


class ExpCutoff(nn.Module):
    def __init__(self, rc: float = 5.0):
        super().__init__()
        self.register_buffer('rc', torch.tensor(rc))

    def forward(self, x: Tensor, inverse: bool = False) -> Tensor:
        x = ops.exp_cutoff(x, self.rc)
        if inverse:
            x = 1 - x
        return x
       

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


class DSequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.module = nn.ModuleList(modules)

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for m in self.module:
            data = m(data)
        return data                


class AtomicShift(nn.Module):
    def __init__(self, key_in: str, key_out: str, num_types: int = 64,
                 dtype: torch.dtype = torch.float, requires_grad: bool = True, reduce_sum=False):
        super().__init__()
        shifts = nn.Embedding(num_types, 1, padding_idx=0, dtype=dtype)
        shifts.weight.requires_grad_(requires_grad)
        self.shifts = shifts
        self.key_in = key_in
        self.key_out = key_out
        self.reduce_sum = reduce_sum

    def extra_repr(self) -> str:
        return f'key_in: {self.key_in}, key_out: {self.key_out}'

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        shifts = self.shifts(data['numbers']).squeeze(-1)
        if self.reduce_sum:
            shifts = nbops.mol_sum(shifts, data)
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
        data[self.key_out] = nbops.mol_sum(data[self.key_in], data)
        return data


class Output(nn.Module):
    def __init__(self, mlp: Union[Dict, nn.Module], n_in: int, n_out: int,
                 key_in: str, key_out: str):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out
        if not isinstance(mlp, nn.Module):
            mlp = MLP(n_in=n_in, n_out=n_out, **mlp)
        self.mlp = mlp

    def extra_repr(self) -> str:
        return f'key_in: {self.key_in}, key_out: {self.key_out}'

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        v = self.mlp(data[self.key_in]).squeeze(-1)
        if data['_input_padded'].item():
            v = nbops.mask_i_(v, data, mask_value=0.0)
        data[self.key_out] = v
        return data
    

class Forces(nn.Module):
    def __init__(self, module: nn.Module, x: str = 'coord', y: str = 'energy', key_out: str = 'forces'):
        super().__init__()
        self.module = module
        self.x = x
        self.y = y
        self.key_out = key_out

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        data[self.x].requires_grad_(True)
        data = self.module(data)
        y = data[self.y]
        g = torch.autograd.grad(
            [y.sum()], [data[self.x]], create_graph=self.training)[0]
        assert g is not None
        data[self.key_out] = - g
        torch.set_grad_enabled(prev)
        return data
    

class Dipole(nn.Module):
    def __init__(self, key_in: str = 'charges',
                 key_out: str = 'dipole',
                 center_coord: bool = False):
        super().__init__()
        self.center_coord = center_coord
        self.key_out = key_out
        self.key_in = key_in
        self.register_buffer('mass', constants.get_masses())

    def extra_repr(self) -> str:
        return f'key_in: {self.key_in}, key_out: {self.key_out}, center_coord: {self.center_coord}'

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        q = data[self.key_in]
        r = data['coord']
        if self.center_coord:
            r = ops.center_coordinates(r, data, self.mass[data['numbers']])
        data[self.key_out] = nbops.mol_sum(q.unsqueeze(-1) * r, data)
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
            r = ops.center_coordinates(r, data, self.mass[data['numbers']])
        _x = torch.cat([r.pow(2), r * r.roll(-1, -1)], dim=-1)
        quad = nbops.mol_sum(q.unsqueeze(-1) * _x, data)
        _x1, _x2 = quad.split(3, dim=-1)
        _x1 = _x1 - _x1.mean(dim=-1, keepdim=True)
        quad = torch.cat([_x1, _x2], dim=-1)
        data[self.key_out] = quad
        return data


class SRRep(nn.Module):
    """GFN1-stype short range repulsion function
    """
    def __init__(self, key_out='e_rep', cutoff_fn='none', rc=5.2, reduce_sum=True):
        super().__init__()
        from aimnet.constants import get_gfn1_rep

        self.key_out = key_out
        self.cutoff_fn = cutoff_fn
        self.reduce_sum = reduce_sum

        self.register_buffer('rc', torch.tensor(rc))
        gfn1_repa, gfn1_repb = get_gfn1_rep()
        weight = torch.stack([gfn1_repa, gfn1_repb], axis=-1)
        self.params = nn.Embedding(87, 2, padding_idx=0, _weight=weight)
        self.params.weight.requires_grad_(False)

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        p = self.params(data['numbers'])
        p_i, p_j = nbops.get_ij(p, data)
        p_ij = p_i * p_j
        alpha_ij, zeff_ij = p_ij.unbind(-1)
        d_ij = data['d_ij']
        e = torch.exp(- alpha_ij * d_ij.pow(1.5)) * zeff_ij / d_ij
        e = nbops.mask_ij_(e, data, 0.0)
        if self.cutoff_fn == 'exp_cutoff':
            e = e * ops.exp_cutoff(d_ij, self.rc)
        elif self.cutoff_fn == 'cosine_cutoff':
            e = e * ops.cosine_cutoff(d_ij, self.rc)
        e = e.sum(-1)
        if self.reduce_sum:
            e = nbops.mol_sum(e, data)
        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e
        else:
            data[self.key_out] = e
        return data
    
    
class LRCoulomb(nn.Module):
    def __init__(self, key_in: str = 'charges', key_out: str = 'e_h',
                 rc: float = 4.6, method='simple',
                 dsf_alpha: float = 0.2, dsf_rc: float = 15.0):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out
        self._factor = constants.half_Hartree * constants.Bohr
        self.register_buffer('rc', torch.tensor(rc))
        self.dsf_alpha = dsf_alpha
        self.dsf_rc = dsf_rc
        if method in ('simple', 'dsf', 'ewald'):
            self.method = method
        else:
            raise ValueError(f'Unknown method {method}')
        
    def coul_simple(self, data: Dict[str, Tensor]) -> Tensor:
        data = ops.lazy_calc_dij_lr(data)
        d_ij = data['d_ij_lr']
        q = data[self.key_in]
        q_i, q_j = nbops.get_ij(q, data, suffix='_lr')
        q_ij = q_i * q_j
        fc = 1.0 - ops.exp_cutoff(d_ij, self.rc)
        e_ij = fc * q_ij / d_ij
        e_ij = nbops.mask_ij_(e_ij, data, 0.0, suffix='_lr')
        e_i = e_ij.sum(-1)
        e = self._factor * nbops.mol_sum(e_i, data)
        return e
    
    def coul_simple_sr(self, data: Dict[str, Tensor]) -> Tensor:
        d_ij = data['d_ij']
        q = data[self.key_in]
        q_i, q_j = nbops.get_ij(q, data)
        q_ij = q_i * q_j
        fc = ops.exp_cutoff(d_ij, self.rc)
        e_ij = fc * q_ij / d_ij
        e_ij = nbops.mask_ij_(e_ij, data, 0.0)
        e_i = e_ij.sum(-1)
        e = self._factor * nbops.mol_sum(e_i, data)
        return e
    
    def coul_dsf(self, data: Dict[str, Tensor]) -> Tensor:
        data = ops.lazy_calc_dij_lr(data)
        d_ij = data['d_ij_lr']
        q = data[self.key_in]
        q_i, q_j = nbops.get_ij(q, data, suffix='_lr')
        epot = ops.coulomb_potential_dsf(q_j, d_ij, self.dsf_rc, self.dsf_alpha, data)
        q_i = q_i.squeeze(-1)
        e = q_i * epot
        e = self._factor * nbops.mol_sum(e, data)
        e = e - self.coul_simple_sr(data)
        return e

    def coul_ewald(self, data: Dict[str, Tensor]) -> Tensor:
        nb_mode = nbops.get_nb_mode(data)
        assert nb_mode == 1, "Ewald is only available in nb_mode 1"
        assert 'cell' in data, "cell is required"
        # single moculele implementation.
        accuracy = 1e-6
        coord, cell, charges = data['coord'][:-1], data['cell'], data['charges'][:-1]
        N = coord.shape[0]
        volume = torch.det(cell)
        eta = ((volume ** 2 / N) ** (1 / 6)) / math.sqrt(2.0 * math.pi)
        cutoff_real = math.sqrt(-2.0 * math.log(accuracy)) * eta
        cutoff_recip = math.sqrt(-2.0 * math.log(accuracy)) / eta

        # real space
        _grad_mode = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        shifts = ops.get_shifts_within_cutoff(cell, cutoff_real)  # (num_shifts, 3)
        torch.set_grad_enabled(_grad_mode)
        disps_ij = coord[None, :, :] - coord[:, None, :]
        disps = disps_ij[None, :, :, :] + torch.matmul(shifts, cell)[:, None, None, :]
        distances_all = torch.linalg.norm(disps, dim=-1)  # (num_shifts, num_atoms, num_atoms)
        within_cutoff = (distances_all > 0.1) & (distances_all < cutoff_real)
        distances = distances_all[within_cutoff]
        e_real_matrix_aug = torch.zeros_like(distances_all)
        e_real_matrix_aug[within_cutoff] = torch.erfc(distances / (math.sqrt(2) * eta)) / distances
        e_real_matrix = e_real_matrix_aug.sum(dim=0)

        # reciprocal space
        recip = 2 * math.pi * torch.transpose(torch.linalg.inv(cell), 0, 1)
        _grad_mode = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        shifts = ops.get_shifts_within_cutoff(recip, cutoff_recip)
        torch.set_grad_enabled(_grad_mode)
        ks_all = torch.matmul(shifts, recip)
        length_all = torch.linalg.norm(ks_all, dim=-1)
        within_cutoff = (length_all > 0.1) & (length_all < cutoff_recip)
        ks = ks_all[within_cutoff]
        length = length_all[within_cutoff]
        # disps_ij[i, j, :] is displacement vector r_{ij}, (num_atoms, num_atoms, 3)
        # disps_ij = coord[None, :, :] - coord[:, None, :] # computed above
        phases = torch.sum(ks[:, None, None, :] * disps_ij[None, :, :, :], dim=-1)
        e_recip_matrix_aug = (
            torch.cos(phases)
            * torch.exp(-0.5 * torch.square(eta * length[:, None, None]))
            / torch.square(length[:, None, None])
        )
        e_recip_matrix = (
            4.0
            * math.pi
            / volume
            * torch.sum(e_recip_matrix_aug, dim=0)
        )
        # self interaction
        device = coord.device
        diag = -math.sqrt(2.0 / math.pi) / eta * torch.ones(N, device=device)
        e_self_matrix = torch.diag(diag)

        energy_matrix = e_real_matrix + e_recip_matrix + e_self_matrix
        e = self._factor * (energy_matrix * charges[:, None] * charges[None, :]).sum()
        e = e - self.coul_simple_sr(data)
        return e

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.method == 'simple':
            e = self.coul_simple(data)
        elif self.method == 'dsf':
            e = self.coul_dsf(data)
        elif self.method == 'ewald':
            e = self.coul_ewald(data)
        else:
            raise ValueError(f'Unknown method {self.method}')
        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e
        else:
            data[self.key_out] = e
        return data


class DispParam(nn.Module):
    def __init__(self, ref_c6: Optional[Union[Tensor, Dict[int, float]]] = None,
                 ref_alpha: Optional[Union[Tensor, Dict[int, float]]] = None,
                 ptfile: Optional[str] = None,
                 key_in='disp_param', key_out='disp_param'):
        super().__init__()
        # load data
        if ptfile is not None:
            ref = torch.load(ptfile)
        else:
            ref = torch.zeros(87, 2)
        for i, p in enumerate([ref_c6, ref_alpha]):
            if p is not None:
                if isinstance(p, Tensor):
                    ref[:p.shape[0], i] = p
                else:
                    for k, v in p.items():
                        ref[k, i] = v
        # alpha=1 for dummy atom
        ref[0, 1] = 1.0
        self.register_buffer('disp_param0', ref)
        self.key_in = key_in
        self.key_out = key_out

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        disp_param_mult = data[self.key_in].clamp(min=-4, max=4).exp()
        disp_param = self.disp_param0[data['numbers']]
        vals = disp_param * disp_param_mult
        data[self.key_out] = vals
        return data
    

class D3TS(nn.Module):
    """ D3-like pairwise dispersion with TS combination rule
    """
    def __init__(self, a1: float, a2: float, s8: float, s6: float = 1.0,
                 key_in='disp_param', key_out='energy'):
        super().__init__()
        self.register_buffer('r4r2', constants.get_r4r2())
        self.a1 = a1
        self.a2 = a2
        self.s6 = s6
        self.s8 = s8
        self.key_in = key_in
        self.key_out = key_out

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        disp_param = data[self.key_in]
        disp_param_i, disp_param_j = nbops.get_ij(disp_param, data, suffix='_lr')
        c6_i, alpha_i = disp_param_i.unbind(dim=-1)
        c6_j, alpha_j = disp_param_j.unbind(dim=-1)

        # TS combination rule
        c6ij = 2 * c6_i * c6_j / (c6_i * alpha_j / alpha_i + c6_j * alpha_i / alpha_j).clamp(min=1e-4)

        rr = self.r4r2[data['numbers']]
        rr_i, rr_j = nbops.get_ij(rr, data, suffix='_lr')
        rrij = 3 * rr_i * rr_j
        rrij = nbops.mask_ij_(rrij, data, 1.0, suffix='_lr')
        r0ij = self.a1 * rrij.sqrt() + self.a2

        ops.lazy_calc_dij_lr(data)
        d_ij = data['d_ij_lr'] * constants.Bohr_inv
        e_ij = c6ij * (self.s6 / (d_ij.pow(6) + r0ij.pow(6)) + self.s8 * rrij / (d_ij.pow(8) + r0ij.pow(8)))
        e = - constants.half_Hartree * nbops.mol_sum(e_ij.sum(-1), data)

        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e
        else:
            data[self.key_out] = e

        return data
    

class DFTD3(nn.Module):
    def __init__(self, s8: float, a1: float, a2: float, s6: float = 1.0,
                 datafile: Optional[str] = None, key_out='energy'):
        super().__init__()
        self.key_out = key_out
        # BJ damping parameters
        self.s6 = s6
        self.s8 = s8
        self.s9 = 4.0 / 3.0
        self.a1 = a1
        self.a2 = a2
        self.a3 = 16.0
        # CN parameters
        self.k1 = - 16.0
        self.k3 = -4.0
        # data
        self.register_buffer('c6ab', torch.zeros(95, 95, 5, 5, 3))
        self.register_buffer('r4r2', torch.zeros(95))
        self.register_buffer('rcov', torch.zeros(95))
        self.register_buffer('cnmax', torch.zeros(95))
        if datafile is None:
            datafile = os.path.join(os.path.dirname(__file__), 'd3bj_data.pt')    
        sd = torch.load(datafile)
        self.load_state_dict(sd)

    def _calc_c6ij(self, data: Dict[str, Tensor]) -> Tensor:
        # CN part
        # short range for CN
        #d_ij = data['d_ij'] * constants.Bohr_inv
        data = ops.lazy_calc_dij_lr(data)
        d_ij = data['d_ij_lr'] * constants.Bohr_inv

        numbers = data['numbers']
        numbers_i, numbers_j = nbops.get_ij(numbers, data, suffix='_lr')
        rcov_i, rcov_j = nbops.get_ij(self.rcov[numbers], data, suffix='_lr')
        rcov_ij = rcov_i + rcov_j
        cn_ij = 1.0 / (1.0 + torch.exp(self.k1 * (rcov_ij / d_ij - 1.0)))
        cn_ij = nbops.mask_ij_(cn_ij, data, 0.0, suffix='_lr')
        cn = cn_ij.sum(-1)
        cn = torch.clamp(cn, max=self.cnmax[numbers]).unsqueeze(-1).unsqueeze(-1)
        cn_i, cn_j = nbops.get_ij(cn, data, suffix='_lr')
        c6ab = self.c6ab[numbers_i, numbers_j]
        c6ref, cnref_i, cnref_j = torch.unbind(c6ab, dim=-1)
        c6ref = nbops.mask_ij_(c6ref, data, 0.0, suffix='_lr')
        l_ij = torch.exp(self.k3 * ((cn_i - cnref_i).pow(2) + (cn_j - cnref_j).pow(2)))
        w = l_ij.flatten(-2, -1).sum(-1)
        z = torch.einsum('...ij,...ij->...', c6ref, l_ij)
        _w = w < 1e-5
        z[_w] = 0.0
        c6_ij = z / w.clamp(min=1e-5)
        return c6_ij

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        c6ij = self._calc_c6ij(data)

        rr = self.r4r2[data['numbers']]
        rr_i, rr_j = nbops.get_ij(rr, data, suffix='_lr')
        rrij = 3 * rr_i * rr_j
        rrij = nbops.mask_ij_(rrij, data, 1.0, suffix='_lr')
        r0ij = self.a1 * rrij.sqrt() + self.a2

        ops.lazy_calc_dij_lr(data)
        d_ij = data['d_ij_lr'] * constants.Bohr_inv
        e_ij = c6ij * (self.s6 / (d_ij.pow(6) + r0ij.pow(6)) + self.s8 * rrij / (d_ij.pow(8) + r0ij.pow(8)))
        e = - constants.half_Hartree * nbops.mol_sum(e_ij.sum(-1), data)

        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e
        else:
            data[self.key_out] = e
        return data
