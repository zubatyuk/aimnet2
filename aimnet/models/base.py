import torch
from torch import nn, Tensor
from typing import Dict, Final
from aimnet import nbops


class AIMNet2Base(nn.Module):
    _required_keys: Final = ['coord', 'numbers', 'charge']
    _required_keys_dtype: Final = [torch.float32, torch.int64, torch.float32]
    _optional_keys: Final = ['mult', 'nbmat', 'nbmat_lr', 'mol_idx', 'shifts', 'cell']
    _optional_keys_dtype: Final = [torch.float32, torch.int64, torch.int64, torch.int64, torch.float32, torch.float32]
    __constants__ = ['_required_keys', '_required_keys_dtype', '_optional_keys', '_optional_keys_dtype']

    def __init__(self):
        super().__init__()

    def _prepare_dtype(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for k, d in zip(self._required_keys, self._required_keys_dtype):
            assert k in data, f"Key {k} is required"
            data[k] = data[k].to(d)
        for k, d in zip(self._optional_keys, self._optional_keys_dtype):
            if k in data:
                data[k] = data[k].to(d)
        return data

    def prepare_input(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """ Some sommon operations
        """
        data = self._prepare_dtype(data)
        data = nbops.set_nb_mode(data)
        data = nbops.calc_masks(data)
        return data
    

