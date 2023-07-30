from typing import Union, Optional, Dict, Any, Iterable, Tuple

import h5py
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
        self.strict = strict
        self.mask = mask
        self._data = self.load_data(data)
        self.assert_strictness()
        self.cow = cow

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
        if self.cow or self._root is None:
            self._data[key] = to_array(value)
        else:
            self._data[key] = self._root.array(key, to_array(value),
                                               overwrite=True)

    def __getitem__(self, item: Union[str, int, slice, ndarray, Tuple[str, ndarray]]) -> Union[
        Any, Dict[str, ndarray]]:
        if isinstance(item, str):
            val = self._data[item]
            if self.mask is not None:
                val = to_array(val)[self.mask]

        elif isinstance(item, Tuple):
            val = self._data[item[0]]
            if self.mask is not None:
                val = to_array(val)[self.mask][item[1]]
            else:
                val = to_array(val)[item[1]]

        else:
            if self.mask is not None:
                val = {k: to_array(v)[self.mask][item] for k, v in self.items()}
            else:
                val = {k: to_array(v)[item] for k, v in self.items()}
        return val

    def __delitem__(self, key):
        del self._data[key]
        if not (self.cow or self._root is None):
            del self._root[key]

    def flush(self):
        if self._root is not None and self.cow:
            for key in self.keys():
                if isinstance(self._data[key], ndarray):
                    self._root.array(key, self._data[key], overwrite=True)
                    self._data[key] = self._root[key]
            for key in self._root.keys():
                if key not in self.keys():
                    del self._root[key]

    def to_memory(self, shard=(0, 1), keys=None):
        if keys is None:
            keys = self.keys()
        else:
            assert set(keys) == set(keys) & set(self.keys())
        self._root = None
        self._data = {k: to_array(self._data[k])[shard[0]::shard[1]] for k in keys}

    def to_root(self, root: zarr.hierarchy.Group, items=None):
        if items is None:
            items = slice(0, len(self))
        for key, value in self.items():
            root.array(key, to_array(value)[items], overwrite=True)
            self._data[key] = root[key]
        self._root = root

        for key in self._root.keys():
            if key not in self.keys():
                del self._root[key]

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
                self._data[k].append(to_array(other._data[k]))

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
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask
        self.assert_strictness()

    @classmethod
    def read_from_h5(cls, file: Union[str, h5py.File, h5py.Group], root=None, keys=None,
            **kwargs):
        if isinstance(file, str):
            f = h5py.File(file, 'r')
        else:
            f = file

        if keys is None:
            keys = f.keys()
        if root is not None:
            data = root
        else:
            data = {}
        instance = cls(data, **kwargs)

        for k in keys:
            instance[k] = np.array(f[k])

        if isinstance(file, str):
            f.close()
        return instance

    def save_h5(self, file: Union[str, h5py.File, h5py.Group], keys=None):
        if isinstance(file, str):
            f = h5py.File(file, 'w')
        else:
            f = file
        if keys is None:
            keys = self.keys()
        for k in keys:
            f.create_dataset(k, data=to_array(self[k]))
        if isinstance(file, str):
            f.close()

    def random_split(self, *fractions, seed=None):
        assert 0 < sum(fractions) <= 1
        assert all(f > 0 for f in fractions)
        idx = np.arange(len(self))
        np.random.seed(seed)
        np.random.shuffle(idx)
        sections = np.around(np.cumsum(fractions) * len(self)).astype(int)
        if sum(fractions) == 1:
            sections = sections[:-1]
        return np.array_split(idx, sections)

    def get_split(self, split, keys=None, root=None, **group_kwargs):
        if keys is None:
            keys = self.keys()
        if isinstance(split, list):

            groups = []

            for i, splt in enumerate(split):
                if isinstance(root, zarr.hierarchy.Group):
                    r = root.create_group(str(i))
                elif isinstance(root, list):
                    r = root[i]
                else:
                    r = None
                groups.append(self.get_split(splt, keys=keys, root=r, **group_kwargs))
            return groups

        elif isinstance(split, ndarray):
            if root is not None:
                data = root
            else:
                data = {}
            datagroup = self.__class__(data, **group_kwargs)

            for k in keys:
                datagroup[k] = self[(k, split)]
            return datagroup

        else:
            raise NotImplementedError

    def apply_peratom_shift(self, sap_dict, key_in='energy', key_out='energy',
            numbers_key='numbers'):
        ntyp = max(sap_dict.keys()) + 1
        sap = np.zeros(ntyp) * np.nan
        for k, v in sap_dict.items():
            sap[k] = v

        val = to_array(self._data[key_in]) - \
              sap[to_array(self._data[numbers_key])].sum(axis=-1)
        self[key_out] = val


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
    import h5py

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
    zarr_group.merge(another_zarr_group, strict=False)

    another_zarr_group.save_h5("toy.h5")
    group = root.create_group("loaded_from_h5")

    loaded_group = DataGroup.read_from_h5("toy.h5", root=group, cow=False)

    print(loaded_group._root.tree())

    dataset = h5py.File("test.h5")

    print(dataset["027"])

    group = root.create_group("test_dataset")
    datagroup = DataGroup.read_from_h5(dataset["027"], root=group, cow=False)

    splits = datagroup.random_split(0.9, 0.1)
    group = root.create_group("test")
    group = root.create_group("train")
    datasets = datagroup.get_split(splits, root=[root["train"], root["test"]], cow=False)
    print(datasets[0]._root.tree())

    datasets = datagroup.get_split(splits, root=None)

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
