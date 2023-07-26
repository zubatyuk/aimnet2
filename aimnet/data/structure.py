import zarr
from typing import Union, Optional, List, Dict, Any, Iterable

from numpy import ndarray

from zarr.hierarchy import Group, Array

class DataGroup:
    def __init__(self,
                data: Union[str,
                            Dict[str, Any],
                            zarr.hierarchy.Group,
                            zarr.storage.Store],
                cow: bool = True,
                mask: Optional[ndarray] = None,
                strict: bool = True, #first dimension unique
            ):
        self._root = None
        self._modified_keys = set()
        self.strict = strict
        self._data = self.load_data(data)
        self.assert_stricness()
        self.cow = cow
        self.mask = mask

        if isinstance(data, str):
            zarr.open_group()

    def load_data(self, data) -> Dict[str, Any]:
        if isinstance(data, str) or isinstance(data, zarr.storage.Store):
            data = zarr.open_group(data)

        if isinstance(data, zarr.hierarchy.Group):
            self._root = data
            data = {k: data[k] for k in data.array_keys()}

        return data


    def assert_strictness(self):
        pass

    def __len__(self):
        pass

    def keys(self):
        pass

    def items(self):
        pass

    def values(self):
        pass

    @staticmethod
    def _to_array(value):
        pass

    def __setitem__(self, key, value):
        pass
        if self.cow or self._root is None:
            self._data[key] = self._to_array(value)
        else:
            if isinstance():
                self._data[key] = self._root.create_arry(self._to_array(value),
                                                         overwrite=True)
            else:
                raise NotImplementedError

    def __getitem__(self, item: Union[int, slice, ndarray]) -> Dict[str, ndarray]:
        if self.mask is not None:
            # apply mask before indexing

        pass

    def __delitem__(self, key):
        pass

    def flush(self):
        #update modified keys
        pass

    def to_memory(self, shard=(0, 1), keys=None):
        self._root = None
        pass

    def to_root(self, root):
        pass

    def merge(self, other, strict=True):
        pass

    def apply_mask(self):
        # if cow FAlse change in init.
        pass

    def set_mask(self):
        pass

class SGDataset:
    def __init__(self,
                data: Union[str,
                            Dict[Union[str, int], Any]] = None, # join groups as it is
                            strict: bool = True,
                            mask: Dict[str, ndarray] = None,
                            shard: tuple(int, int) = (0, 1), #
                            keys: Iterable[int] = None,

                                       ):

    def __len__(self, ):
        pass

    def __getitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass



