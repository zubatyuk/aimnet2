import torch
from torch import Tensor
from typing import Dict, Any
from aimnet.config import get_module
from functools import partial


class MTLoss:
    """ Multi-target loss function with fixed weights.

    This class allows for the combination of multiple loss functions, each with a specified weight.
    The weights are normalized to sum to 1. The loss functions are applied to the predictions and 
    true values, and the weighted sum of the losses is computed.

    Attributes:
        weights (Tensor): Normalized weights for each loss function.
        functions (List[Dict]): List of infividual loss functions definitions.
        
    Loss functions definition must contain keys:
        name (str): The name of the loss function.
        fn (str): The loss function (e.g. `aimnet2.train.loss.mse_loss_fn`).
        weight (float): The weight of the loss function.
        kwargs (Dict): Optional, additional keyword arguments for the loss function.

    Methods:
        __call__(y_pred, y_true):
            Computes the weighted sum of the losses from the individual loss functions.
            Args:
                y_pred (Dict[str, Tensor]): Predicted values.
                y_true (Dict[str, Tensor]): True values.
            Returns:
                Dict[str, Tensor]: total loss under key 'loss' and values for individual components.
    """

    def __init__(self, components: Dict[str, Any]):
        w_sum = sum(c['weight'] for c in components.values())
        self.components = dict()
        for name, c in components.items():
            kwargs = c.get('kwargs', dict())
            fn = partial(get_module(c['fn']), **kwargs)
            self.components[name] = (fn, c['weight'] / w_sum)

    def __call__(self, y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor]) -> Dict[str, Tensor]:
        loss = dict()
        for name, (fn, w) in self.components.items():
            l = fn(y_pred=y_pred, y_true=y_true)
            loss[name] = l * w
        # special name for the total loss
        loss['loss'] = sum(loss.values())
        return loss


def mse_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str, key_true: str) -> Tensor:
    """ General MSE loss function
    """
    x = y_true[key_true]
    y = y_pred[key_pred]
    l = torch.nn.functional.mse_loss(x, y)
    return l


def peratom_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str, key_true: str) -> Tensor:
    """ MSE loss function with per-atom normalization correction.
    Suitable when some of the values are zero both in y_pred and y_true due to padding of inputs.
    """
    x = y_true[key_true]
    y = y_pred[key_pred]
    l = torch.nn.functional.mse_loss(x, y)

    if '_natom' in y_pred and y_pred['_natom'].numel() > 1:
        l = l * _natom_scaling(y_pred)

    return l

def energy_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str = 'energy', key_true: str = 'energy') -> Tensor:
    """MSE loss normalized by the number of atoms.
    """
    x = y_true[key_true]
    y = y_pred[key_pred]
    s = y_pred['_natom'].sqrt()
    if y_pred['_natom'].numel() > 1:
        l = ((x - y).pow(2) / s).mean()
    else:
        l = torch.nn.functional.mse_loss(x, y) / s
    return l

def fuzzy_charges_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str = 'charges', key_true: str = 'charges', eta=1.0) -> Tensor:
    x = y_true[key_true]
    y = y_pred[key_pred]
    d_ij = y_pred['d_ij'].clamp(max=6.0)
    with torch.no_grad():
        w = torch.exp(- eta * d_ij.pow(2))
        w /= w.sum(-1, keepdim=True).clamp(min=1e-6)
    x = (x.unsqueeze(-2) * w).sum(-1)
    y = (y.unsqueeze(-2) * w).sum(-1)

    l = torch.nn.functional.mse_loss(x, y)

    if '_natom' in y_pred and y_pred['_natom'].numel() > 1:
        l = l * _natom_scaling(y_pred)

    return l

def dipole_charge_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str = 'dipole', key_true: str = 'dipole', correct_for_extent=True) -> Tensor:
    x = y_true[key_true]
    y = y_pred[key_pred]
    if correct_for_extent:
        spatial_extent = y_pred['coord'].detach().pow(2).sum(-2)
        l = ((x - y).pow(2) / spatial_extent.clamp(min=1e-3)).mean()
    else:
        l = torch.nn.functional.mse_loss(x, y)

    if '_natom' in y_pred and y_pred['_natom'].numel() > 1:
        l = l * _natom_scaling(y_pred)

    return l


def quadrupole_charge_loss_fn(y_pred: Dict[str, Tensor], y_true: Dict[str, Tensor], key_pred: str = 'quadrupole', key_true: str = 'quadrupole', correct_for_extent=True) -> Tensor:
    x = y_true[key_true]
    y = y_pred[key_pred]
    if correct_for_extent:
        coord = y_pred['coord'].detach()
        spatial_extent2 = torch.cat([coord.pow(2), coord * coord.roll(1, -1)], dim=-1).pow(2).sum(-2)
        l = ((x - y).pow(2) / spatial_extent2.clamp(min=1e-3)).mean()
    else:
        l = torch.nn.functional.mse_loss(x, y)

    if '_natom' in y_pred and y_pred['_natom'].numel() > 1:
        l = l * _natom_scaling(y_pred)

    return l

def _natom_scaling(y_pred: Dict[str, Tensor]) -> float:
    """MSE loss normalization 
    """
    factor = y_pred['numbers'].numel() / y_pred['_natom'].sum().item()
    return factor
