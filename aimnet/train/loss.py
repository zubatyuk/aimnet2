import torch
from torch import jit, Tensor
from typing import List, Dict
from aimnet.config import get_module
from functools import partial


class MTLoss:
    def __init__(self, components: List[Dict]):
        weights = []
        functions = []
        for c in components:
            kwargs = c.get('kwargs', dict())
            fn = partial(get_module(c['fn']), **kwargs)
            functions.append(fn)
            weights.extend(c['weights'])
        self.weights = torch.tensor(weights) / sum(weights)
        self.functions = functions

    def __call__(self, y_pred, y_true):
        loss = []
        for fn in self.functions:
            _l = fn(y_pred, y_true)
            if not _l.ndim:
                _l = _l.unsqueeze(0)
            loss.append(_l)
        self.weights = self.weights.to(loss[0].device)
        loss = (torch.cat(loss) * self.weights).sum()
        #if loss > 0.5:
        #    import os
        #    d = dict(y_pred=y_pred, y_true=y_true)
        #    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', '0'))
        #    torch.save(d, f'batch_{LOCAL_RANK}.pt')
        #    abba
        #print('>>> total', loss.item())
        return loss


@jit.script
def c6ij_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]) -> Tensor:
    _b, _n = y_pred['numbers'].shape
    mask = torch.ones(_b, _n, _n, dtype=torch.bool, device=y_pred['c6ij'].device).tril_()
    c6ij_true = y_true['c6ij'][mask]
    c6ij_pred = y_pred['c6ij'][mask]
    loss = (1 - c6ij_pred / c6ij_true).pow(2).mean()
    return loss


@jit.script
def energy_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]) -> Tensor:
    s = y_pred['_natom'].sqrt()
    l = ((y_pred['energy'] - y_true['energy']).pow(2) / s).mean(dim=-1)
    #l = (y_pred['energy'] - y_true['energy']).pow(2).mean(dim=-1)
    #print('>>', l.item())
    return l


@jit.script
def dipole_charge_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]) -> Tensor:
    true = y_true['dipole'].unsqueeze(0)
    pred = y_pred['dipole']  # B, 3
    l = (true - pred).pow(2).flatten(1, -1).mean(dim=-1)
    return l
    # extent = y_true['spatial_extent']  # B, 3
    #return ((true - pred).pow(2) / extent).flatten(1, -1).mean(dim=-1)

@jit.script
def quadrupole_charge_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]) -> Tensor:
    true = y_true['quadrupole'].unsqueeze(0)
    pred = y_pred['quadrupole']
    return (true - pred).pow(2).flatten(1, -1).mean(dim=-1)  
    #extent = y_true['spatial_extent2']
    #return ((true - pred).pow(2) / extent).flatten(1, -1).mean(dim=-1)  

@jit.script
def forces_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]) -> Tensor:
    #if 'forces' in y_pred:
    #    _natom = y_pred['_natom']
    #    x = y_true['forces']
    #    y = y_pred['forces']
    #    l = torch.nn.functional.mse_loss(x, y)
    #    if _natom.numel() > 1:
    #        l = l * (x.shape[0] * x.shape[1] / _natom.sum())
    #else:
    l = torch.tensor(0.0, device=y_true['forces'].device)
    _natom = y_pred['_natom']
    x = y_true['forces']
    y = y_pred['forces']
    l = torch.nn.functional.mse_loss(x, y)
    if _natom.numel() > 1:
        l = l * (x.shape[0] * x.shape[1] / _natom.sum())
    return l

@jit.script
def c6i_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]) -> Tensor:
    _natom = y_pred['_natom']
    x = y_true['c6i']
    y = y_pred['c6i']
    err = (1.0 - (y / x.clamp(min=0.1))).pow(2)
    if y_pred['pad_mask'].numel() > 1:
        err[y_pred['pad_mask']] = 0.0
    l = err.mean()
    if _natom.numel() > 1:
        l = l * (x.shape[0] * x.shape[1] / _natom.sum())
    return l

@jit.script
def alpha_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]) -> Tensor:
    _natom = y_pred['_natom']
    x = y_true['alpha']
    y = y_pred['alpha']
    err = (1.0 - (y / x.clamp(min=0.1))).pow(2)
    if y_pred['pad_mask'].numel() > 1:
        err[y_pred['pad_mask']] = 0.0
    l = err.mean()
    if _natom.numel() > 1:
        l = l * (x.shape[0] * x.shape[1] / _natom.sum())
    return l

