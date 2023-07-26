from typing import Union, Optional, Dict, Any, Iterable, Tuple

import numpy as np
import zarr
from numpy import ndarray


def to_array(value):
    if isinstance(value, zarr.hierarchy.Array):
        value = value[:]
    return value


class DataGroup:
    def __init__(self,
            data: Union[str,
                        Dict[str, Any],
                        zarr.hierarchy.Group,
                        zarr.storage.Store],
            cow: bool = True,
            mask: Optional[ndarray] = None,
            strict: bool = True,
    ):
        self._root = None
        self._modified_keys = set()
        self.strict = strict
        self.mask = mask
        self._data = self.load_data(data)
        self.assert_strictness()
        self.cow = cow

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
        if self.strict:
            n = None
            for key in self._data:
                if n is None:
                    n = len(self._data[key])
                else:
                    assert n == len(self._data[key])
            if n is not None and self.mask is not None:
                assert n == len(self.mask)

    def __len__(self):
        if self.mask is None:
            length = len(self._data[next(iter(self._data.keys()))]) if len(self._data) else 0
        else:
            length = np.sum(self.mask)
        return length

    def keys(self):
        keys = list(sorted(self._data.keys()))
        return keys

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def __setitem__(self, key, value):
        self._modified_keys.add(key)
        if self.cow or self._root is None:
            self._data[key] = to_array(value)
        else:
            self._data[key] = self._root.array(key, to_array(value),
                                               overwrite=True)

    def __getitem__(self, item: Union[str, int, slice, ndarray]) -> Union[Any, Dict[str, ndarray]]:
        if isinstance(item, str):
            val = self._data[item]
            if self.mask is not None:
                val = to_array(val)[self.mask]

        else:
            if self.mask is not None:
                val = {k: to_array(v)[self.mask][item] for k, v in self.items()}
            else:
                val = {k: to_array(v)[item] for k, v in self.items()}
        return val

    def __delitem__(self, key):
        if isinstance(self._data[key], zarr.hierarchy.Array):
            self._modified_keys.add(key)
        del self._data[key]
        if not (self.cow or self._root is None):
            del self._root[key]

    def flush(self):
        if self._root is not None:
            for key in self._modified_keys:
                if key in self._data:
                    self._data[key] = self._root.array(key, self._data[key],
                                                       overwrite=True)
                else:
                    del self._root[key]
        self._modified_keys = set()

    def to_memory(self, shard=(0, 1), keys=None):
        if keys is None:
            keys = self.keys()
        else:
            assert set(keys) == set(keys) & set(self.keys())
        self._root = None
        self._data = {k: to_array(self._data[k])[shard[0]::shard[1]] for k in keys}

    def to_root(self, root: zarr.hierarchy.Group):
        for key in root.keys():
            del root[key]
        for key, value in self.items():
            root.array(key, value)
            self._data[key] = root[key]
        self._root = root

    def merge(self, other, strict=True):
        self.assert_strictness()
        self_keys = set(self.keys())
        other_keys = set(other.keys())

        if strict:
            assert self_keys == other_keys

        keys = self_keys & other_keys
        for k in self_keys - other_keys:
            del self[k]
        for k in keys:
            if self.cow or self._root is None or isinstance(self._data[k], ndarray):
                self._data[k] = np.concatenate([to_array(self._data[k]),
                                                to_array(other._data[k])],
                                               axis=0)
            else:
                self._data[k].append(to_array(other))

        if self.mask is not None:
            if other.mask is None:
                other_mask = np.ones(len(other), dtype=bool)
            else:
                other_mask = other.mask
            self.mask = np.concatenate([self.mask, other_mask])
        self.assert_strictness()

    def apply_mask(self):
        for k, v in self.items():
            val = v[self.mask]
            if self.cow or self._root is None or isinstance(self._data[k], ndarray):
                self._data[k] = val
            else:
                self._root.array(k, val, overwrite=True)
                self._data[k] = self._root[k]

    def set_mask(self, mask):
        self.mask = mask
        self.assert_strictness()


class SGDataset:
    def __init__(self,
            data: Union[str,
                        Dict[Union[str, int], Any]] = None,  # join groups as it is
            strict: bool = True,
            mask: Dict[str, ndarray] = None,
            shard: Tuple[int, int] = (0, 1),  #
            keys: Iterable[int] = None,

    ):
        pass

    def __len__(self, ):
        pass

    def __getitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass


if __name__ == "__main__":
    root = zarr.group()

    group = root.create_group("group_1")

    item = {"species": np.random.randint(0, 10, (10, 30)),
            "forces": np.random.random((10, 30, 3))}

    zarr_group = DataGroup(item, cow=False)
    zarr_group["new_species"] = zarr_group["species"]

    print(zarr_group["new_species"].shape)

    zarr_group.to_root(group)

    print(zarr_group["new_species"])
    print(zarr_group._root.tree())
    zarr_group.to_memory(keys=("forces", "species"), shard=(0, 1))

    print(zarr_group["species"].shape)

    zarr_group.to_root(group)
    zarr_group["other_species"] = zarr_group["species"]
    print(zarr_group._root.tree())

    mask = np.random.randint(0, 1, 10).astype(bool)
    zarr_group.set_mask(mask)

    another_group = root.create_group("group_2")

    item = {"indices": np.random.randint(0, 10, (10, 30)),
            "forces": np.random.random((10, 30, 3)),
            "another_forces": np.random.random((10, 30, 3))}

    another_zarr_group = DataGroup(item)

    print(zarr_group["forces"].shape)
    print(zarr_group._root, zarr_group.cow)
    zarr_group._data
    zarr_group.merge(another_zarr_group, strict=False)

    #
    # zarr_group.cat(another_zarr_group)
    # print(len(zarr_group))
    # print(len(another_zarr_group))
    #
    # another_zarr_group["other_forces"] = np.random.random((10, 30, 3))
    # try:
    #     zarr_group.merge(another_zarr_group)
    # except Exception as e:
    #     print(e)
    #
    # zarr_group.merge(another_zarr_group, strict=False)
