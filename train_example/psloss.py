import torch
from torch import jit, Tensor
from typing import List, Dict
from aimnet.config import get_module
from functools import partial


class PerSampleMTLoss:
    def __init__(self, components: List[Dict]):
        weights = []
        functions = []
        for c in components:
            kwargs = c.get('kwargs', dict())
            fn = partial(get_module(c['fn']), **kwargs)
            functions.append(fn)
            weights.extend(c['weights'])
        self.weights = torch.tensor(weights) / sum(weights)
        self.weights = self.weights.unsqueeze(-1)
        self.functions = functions

    def __call__(self, y_pred, y_true):
        loss = []
        for fn in self.functions:
            _l = fn(y_pred, y_true)
            if _l.ndim == 1:
                _l = _l.unsqueeze(0)
            loss.append(_l)
        self.weights = self.weights.to(loss[0].device)
        loss = (torch.cat(loss) * self.weights).sum(0)
        return loss


@jit.script
def charges_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]) -> Tensor:
    x = y_true['charges']
    y = y_pred['charges']
    diff = x - y
    l = diff.pow(2).mean(-1)
    _natom = y_pred['_natom']
    if _natom.numel() > 1:
        l = l * x.shape[1] / _natom
    return l
    
@jit.script
def energy_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]) -> Tensor:
    s = y_pred['_natom'].sqrt()
    l = (y_pred['energy'] - y_true['energy']).pow(2) / s
    return l

@jit.script
def forces_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]) -> Tensor:
    _natom = y_pred['_natom']
    x = y_true['forces']
    y = y_pred['forces']
    diff = x - y
    l = diff.pow(2).flatten(-2, -1).mean(-1)
    _natom = y_pred['_natom']
    if _natom.numel() > 1:
        l = l * x.shape[1] / _natom
    return l

@jit.script
def c6i_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]) -> Tensor:
    _natom = y_pred['_natom']
    x = y_true['c6i']
    y = y_pred['c6i']
    diff = x - y
    l = diff.pow(2).mean(-1)
    if _natom.numel() > 1:
        l = l * x.shape[1] / _natom
    return l

@jit.script
def alpha_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]) -> Tensor:
    _natom = y_pred['_natom']
    x = y_true['alpha']
    y = y_pred['alpha']
    diff = x - y
    l = diff.pow(2).mean(-1)
    if _natom.numel() > 1:
        l = l * x.shape[1] / _natom
    return l

