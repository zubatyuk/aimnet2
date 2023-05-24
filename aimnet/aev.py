import torch
from torch import nn, Tensor
from typing import List, Optional, Union, Dict
from aimnet import ops
import math


class AEVSV(nn.Module):
    def __init__(self, rmin: float = 0.8, rc_s: float = 5.0,
                 nshifts_s: int = 16, eta_s: Optional[float] = None,
                 rc_v: Optional[float] = None, nshifts_v: Optional[int] = None, eta_v: Optional[float] = None,
                 shifts_s: Optional[List[float]] = None,
                 shifts_v: Optional[List[float]] = None):
        super().__init__()

        self._init_basis(rc_s, eta_s, nshifts_s, shifts_s, rmin, mod='_s')
        if rc_v is not None:
            assert rc_v <= rc_s
            assert nshifts_v is not None
            self._init_basis(rc_v, eta_v, nshifts_v, shifts_v, rmin, mod='_v')
            self._dual_basis = True
        else:
            # dummy init
            self._init_basis(rc_s, eta_s, nshifts_s, shifts_s, rmin, mod='_v')
            self._dual_basis = False

        self.dmat_fill = rc_s ** 2 #20.0 

    def _init_basis(self, rc, eta, nshifts, shifts, rmin, mod='_s'):
        self.register_parameter('rc'+mod, nn.Parameter(
            torch.tensor(rc, dtype=torch.float), requires_grad=False))
        if eta is None:
            eta = (1 / ((rc - rmin) / nshifts)) ** 2
        self.register_parameter('eta'+mod, nn.Parameter(
            torch.tensor(eta, dtype=torch.float), requires_grad=False))
        if shifts is None:
            shifts = torch.linspace(rmin, rc, nshifts + 1)[:nshifts]
        else:
            shifts = torch.as_tensor(shifts, dtype=torch.float)
        self.register_parameter('shifts'+mod, nn.Parameter(
            shifts, requires_grad=False))

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        coord = data['coord']
        r_ij = coord.unsqueeze(-3) - coord.unsqueeze(-2)
        d_ij2 = r_ij.pow(2).sum(-1)

        if 'pad_mask' in data and data['pad_mask'].numel() > 1:
            pad_mask = data['pad_mask']
            dmat_mask = (pad_mask.unsqueeze(-2) + pad_mask.unsqueeze(-1)) | \
                torch.eye(d_ij2.shape[1], device=d_ij2.device,
                          dtype=torch.bool).unsqueeze(0)
        else:
            dmat_mask = torch.eye(
                d_ij2.shape[1], device=d_ij2.device, dtype=torch.bool).unsqueeze(0)
        d_ij2 = d_ij2.masked_fill(dmat_mask, self.dmat_fill)
        d_ij = d_ij2.sqrt()
        data['d_ij'] = d_ij
        data['u_ij'], data['gs'], data['gv'] = self._calc_aev(r_ij, d_ij)
        return data

    def _calc_aev(self, r_ij, d_ij):
        fc_ij = ops.cosine_cutoff(d_ij, self.rc_s)
        gs = ops.exp_expand(d_ij, self.shifts_s,
                            self.eta_s) * fc_ij.unsqueeze(-1)
        u_ij = r_ij / d_ij.unsqueeze(-1)
        if self._dual_basis:
            fc_ij = ops.cosine_cutoff(d_ij, self.rc_v)
            gsv = ops.exp_expand(d_ij, self.shifts_v,
                                 self.eta_v) * fc_ij.unsqueeze(-1)
            gv = gsv.unsqueeze(-1) * u_ij.unsqueeze(-2)
        else:
            gv = gs.unsqueeze(-1) * u_ij.unsqueeze(-2)
        return u_ij, gs, gv


class AEVSV_NB(AEVSV):
    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        coord_i = data['coord']
        _s0, _s1 = data['idx_j'].shape[0], data['idx_j'].shape[1]
        coord_j = torch.index_select(coord_i, 0, data['idx_j'].flatten()).view(_s0, _s1, 3)
        # coord_j = coord_i[data['idx_j']]
        if 'shifts' in data:
            shifts = data['shifts'] @ data['cell']
            coord_j = coord_j + shifts
        r_ij = coord_j - coord_i.unsqueeze(-2)
        d_ij2 = r_ij.pow(2).sum(-1)
        #if 'nb_pad_mask' in data:
        d_ij2 = d_ij2.masked_fill(data['nb_pad_mask'], self.dmat_fill)
        d_ij = d_ij2.sqrt()
        data['d_ij'] = d_ij
        data['u_ij'], data['gs'], data['gv'] = self._calc_aev(r_ij, d_ij)
        return data


class ConvSV(nn.Module):
    def __init__(self, nshifts_s: int, nchannel: int, d2features: bool = False,
                 do_vector: bool = True, nshifts_v: Optional[int] = None,
                 ncomb_v: Optional[int] = None):
        super().__init__()
        nshifts_v = nshifts_v or nshifts_s
        ncomb_v = ncomb_v or nshifts_v
        agh = _init_ahg(nchannel, nshifts_v, ncomb_v)
        # agh = torch.randn(nchannel, nshifts_v, ncomb_v) / nshifts_v
        self.register_parameter('agh', nn.Parameter(agh))
        self.do_vector = do_vector
        self.nchannel = nchannel
        self.d2features = d2features
        self.nshifts_s = nshifts_s
        self.nshifts_v = nshifts_v
        self.ncomb_v = ncomb_v

    def output_size(self):
        n = self.nchannel * self.nshifts_s
        if self.do_vector:
            n += self.nchannel * self.ncomb_v
        return n

    def forward(self, a: Tensor, gs: Tensor, gv: Optional[Tensor] = None) -> Tensor:
        avf = []
        if self.d2features:
            avf_s = torch.einsum('...nmg,...mag->...nag', gs, a)
        else:
            avf_s = torch.einsum('...nmg,...ma->...nag', gs, a)
        avf.append(avf_s.flatten(-2, -1))

        if self.do_vector:
            assert gv is not None
            agh = self.agh
            if self.d2features:
                avf_v = torch.einsum('...nmgd,...mag,agh->...nahd', gv, a, agh)
            else:
                avf_v = torch.einsum('...nmgd,...ma,agh->...nahd', gv, a, agh)
            avf.append(avf_v.pow(2).sum(-1).flatten(-2, -1))

        return torch.cat(avf, dim=-1)


class ConvSV_NB(nn.Module):
    def __init__(self, nshifts_s: int, nchannel: int, d2features: bool = False,
                 do_vector: bool = True, nshifts_v: Optional[int] = None,
                 ncomb_v: Optional[int] = None):
        super().__init__()
        nshifts_v = nshifts_v or nshifts_s
        ncomb_v = ncomb_v or nshifts_v
        agh = torch.randn(nchannel, nshifts_v, ncomb_v) / nshifts_v
        self.register_parameter('agh', nn.Parameter(agh))
        self.do_vector = do_vector
        self.nchannel = nchannel
        self.d2features = d2features
        self.nshifts_s = nshifts_s
        self.nshifts_v = nshifts_v
        self.ncomb_v = ncomb_v

    def output_size(self):
        n = self.nchannel * self.nshifts_s
        if self.do_vector:
            n += self.nchannel * self.ncomb_v
        return n

    def forward(self, a: Tensor, gs: Tensor, gv: Optional[Tensor] = None) -> Tensor:
        avf = []
        if self.d2features:
            avf_s = torch.einsum('nmg,nmag->nag', gs, a)
        else:
            avf_s = torch.einsum('nmg,nma->nag', gs, a)
        avf.append(avf_s.flatten(-2, -1))

        if self.do_vector:
            assert gv is not None
            agh = self.agh
            if self.d2features:
                avf_v = torch.einsum('nmgd,nmag,agh->nahd', gv, a, agh)
            else:
                avf_v = torch.einsum('nmgd,nma,agh->nahd', gv, a, agh)
            avf.append(avf_v.pow(2).sum(-1).flatten(-2, -1))

        return torch.cat(avf, dim=-1)        


def _init_ahg(b, m, n):
    ret = torch.zeros(b, m, n)
    for i in range(b):
        ret[i] = _init_ahg_one(m, n).t()
    return ret


def _init_ahg_one(n, m):
    # make x8 times more vectors to select most diverse
    x = torch.arange(n).unsqueeze(0)
    a1, a2, a3, a4 = torch.randn(8 * m, 4).unsqueeze(-2).unbind(-1)
    y = a1 * torch.sin(a2 * 2 * x * math.pi / n) + a3 * torch.cos(a4 * 2 * x * math.pi / n)
    y -= y.mean(dim=-1, keepdim=True)
    y /= y.std(dim=-1, keepdim=True)
    
    dmat = torch.cdist(y, y)
    # most distant point
    ret = torch.zeros(m, n)
    mask = torch.ones(y.shape[0], dtype=torch.bool)
    i = dmat.sum(-1).argmax()
    ret[0] = y[i]
    mask[i] = False
    
    # simple maxmin impementation
    for j in range(1, m):
        mindist, _ = torch.cdist(ret[:j], y).min(dim=0)
        #maxidx = torch.where(mindist[mask].max() == mindist)[0]
        maxidx = torch.argsort(mindist)[mask][-1]
        assert mask[maxidx] == True
        ret[j] = y[maxidx]
        mask[maxidx] = False
    return ret
    
