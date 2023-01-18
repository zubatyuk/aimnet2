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
            loss.append(_l)
        self.weights = self.weights.to(loss[0].device)
        loss = (torch.stack(loss) * self.weights).sum(0)
        return loss


@jit.script
def peratom_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str, key_true: str) -> Tensor:
    x = y_true[key_true]
    y = y_pred[key_pred]
    diff = x - y
    l = diff.pow(2).mean(-1)
    _natom = y_pred['_natom']
    if _natom.numel() > 1:
        l = l * x.shape[1] / _natom
    return l


@jit.script
def pervalue_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str, key_true: str) -> Tensor:
    x = y_true[key_true]
    y = y_pred[key_pred]
    diff = x - y
    l = diff.view(diff.shape[0], -1).pow(2).mean(-1)
    return l    
    
    
@jit.script
def energy_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str = 'energy', key_true: str = 'energy') -> Tensor:
    s = y_pred['_natom'].sqrt()
    l = (y_pred[key_pred] - y_true[key_true]).pow(2) / s
    return l


@jit.script
def forces_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str = 'forces', key_true: str = 'forces') -> Tensor:
    _natom = y_pred['_natom']
    x = y_true[key_true]
    y = y_pred[key_pred]
    diff = x - y
    l = diff.pow(2).flatten(-2, -1).mean(-1)
    _natom = y_pred['_natom']
    if _natom.numel() > 1:
        l = l * x.shape[1] / _natom
    return l
