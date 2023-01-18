import torch
from torch import Tensor
from typing import Dict
import math


def quad_eigh(v: Tensor) -> Tensor:
    xx, xy, yy, xz, yz, zz = v.unbind(-1)
    xy2, xz2, yz2 = torch.stack([xy, xz, yz], dim=0).pow(2).unbind(dim=0)
    xxyy = xx * yy
    l1 = xx + yy + zz
    l2 = xxyy + yy * zz + xx * zz - xy2 - yz2 - xz2
    l3 = xxyy * zz + 2 * xz * xy * yz - xz2 * yy - xx * yz2 - xy2 * zz
    return torch.stack([l1, l2, l3], dim=-1)


def cosine_cutoff(d: Tensor, rc: Tensor) -> Tensor:
    fc = (d / rc).clamp(0, 1)
    fc = 0.5 * (math.pi * fc).cos() + 0.5
    return fc


def exp_cutoff(d: Tensor, rc: Tensor) -> Tensor:
    fc = torch.exp(-1.0 / (1.0 - (d/rc).clamp(0, 1.0-1e-6).pow(2))
                   ) / 0.36787944117144233
    return fc


def exp_expand(d: Tensor, shifts: Tensor, eta: Tensor) -> Tensor:
    return torch.exp(-eta * (d.unsqueeze(-1) - shifts).pow(2))


def exp_expand_cutoff(d: Tensor, shifts: Tensor, eta: Tensor, rc: Tensor) -> Tensor:
    return torch.exp(-1.0 / (1.0 - (d / rc).pow(2).clamp(max=1-1e-4)) - eta * (d - shifts).pow(2))


def triu_mask(n: int, device: torch.device, diagonal: int = 1) -> torch.Tensor:
    mask = torch.ones(n, n, device=device, dtype=torch.bool)
    mask = torch.triu(mask, diagonal=diagonal)
    return mask


def nqe(Q, q_u, f):
    f = f.pow(2)
    if f.ndim > 1:
        Q_u = q_u.sum(-1, keepdim=True)
        #F_s = f.sum(-1, keepdim=True).clamp(min=1e-4)
        F_s = f.sum(-1, keepdim=True) + 1e-6
        q = q_u + f * (Q.unsqueeze(-1) - Q_u) / F_s
    else:
        Q_u = q_u.sum()
        #F_s = f.sum().clamp(min=1e-4)
        F_s = f.sum() + 1e-6
        q = q_u + f * (Q - Q_u) / F_s
    return q


def nse(Q, q_u, f):
    f = f.pow(2)
    if f.ndim > 2:
        Q_u = q_u.sum(-2, keepdim=True)
        F_s = f.sum(-2, keepdim=True) + 1e-6
        q = q_u + f * (Q.unsqueeze(-2) - Q_u) / F_s
    else:
        Q_u = q_u.sum(0)
        F_s = f.sum(0) + 1e-6
        q = q_u + f * (Q - Q_u) / F_s
    return q
    

def calc_pad_mask(data: Dict[str, Tensor]) -> Dict[str, Tensor]:
    numbers = data['numbers']
    mask = numbers == 0
    if mask.any():
        data['pad_mask'] = mask
        data['_natom'] = (~mask).sum(dim=-1).to(torch.float)
    else:
        data['pad_mask'] = torch.tensor([0], device=numbers.device)
        data['_natom'] = torch.tensor(
            numbers.shape[1], dtype=torch.float, device=numbers.device).unsqueeze(0)
    return data

