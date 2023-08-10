import os
from collections import defaultdict
from copy import deepcopy
from glob import glob
from typing import Union, Dict, Any, Tuple, Sequence, Optional, Iterable

import h5py
import numpy as np
import zarr
from numpy import ndarray
from torch.utils.data.dataloader import DataLoader, default_collate


def to_array(value, copy=False):
    if isinstance(value, zarr.hierarchy.Array):
        value = value[:]

    elif isinstance(value, h5py.Dataset):
        value = value[()]

    elif isinstance(value, ndarray):
        if copy:
            value = value.copy()
    else:
        raise ValueError(f"Invalid data type {type(value)}")

    return value


def clean_group(root: Union[zarr.hierarchy.Group, h5py.Group, None]):
    if isinstance(root, list):
        [clean_group(i) for i in root]
    elif root is not None:
        for k in root.keys():
            del root[k]


def _squeeze(t):
    for d in t:
        for v in d.values():
            v.squeeze_(0)
    return t


def collate_fn(x):
    return _squeeze(default_collate(x))


def _get_keys(object_keys, user_keys=None, strict=True):
    if user_keys is None:
        return object_keys
    else:
        object_keys = set(object_keys)
        user_keys = set(user_keys)
        user_keys = user_keys & object_keys
        if strict:
            assert user_keys & object_keys == user_keys
        return user_keys


class DataGroup:
    def __init__(self,
            data: Union[str,
                        Dict[str, Any],
                        zarr.hierarchy.Group,
                        zarr.storage.Store],
            cow: bool = True,
            strict: bool = True,
            keys=None,
            **kwargs
    ):
        self._root = None
        self.strict = strict
        self._data = self.load_data(data, keys)
        self.assert_strictness()
        self.cow = cow

    def load_data(self, data, keys) -> Dict[str, Any]:
        if isinstance(data, str) or isinstance(data, zarr.storage.Store):
            data = zarr.convenience.open(data)
        if data is None:
            data = {}

        keys = _get_keys(data.keys(), keys)
        if isinstance(data, zarr.hierarchy.Group):
            self._root = data
            data = {k: data[k] for k in keys}
        return data

    def assert_strictness(self):
        if self.strict:
            n = None
            for key in self._data:
                if n is None:
                    n = len(self._data[key])
                else:
                    assert n == len(self._data[key])

    def __len__(self):
        return len(self._data[next(iter(self._data.keys()))]) if len(self._data) else 0

    def __contains__(self, value):
        return value in self.keys()

    def keys(self):
        keys = list(sorted(self._data.keys()))
        return keys

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def pop(self, key):
        val = to_array(self[key])
        del self[key]
        return val

    def rename_key(self, old, new):
        # todo: find a vay how rename an array without copying
        self[new] = to_array(self._data[old])
        del self[old]

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

        elif isinstance(item, Tuple):
            val = self._data[item[0]]
            val = to_array(val)[item[1]]

        else:
            val = {k: to_array(v)[item] for k, v in self.items()}
        return val

    def __delitem__(self, key):
        del self._data[key]
        if not (self.cow or self._root is None):
            del self._root[key]

    def flush(self, keys=None):
        if self._root is not None:
            cow = self.cow
            self.cow = False
            changed_keys = set(self.keys())
            deleted_keys = set(self._root.keys()) - changed_keys
            if keys is not None:
                changed_keys = changed_keys & set(keys)
                deleted_keys = deleted_keys & set(keys)

            for key in changed_keys:
                self[key] = self[key]

            for key in deleted_keys:
                del self._root[key]
            self.cow = cow

    def to_memory(self, keys=None):
        keys = _get_keys(self.keys(), keys)
        self._root = None
        for k in keys:
            self[k] = self[k]

    def get_shard(self, idx=0, size=1, keys=None):
        keys = _get_keys(self.keys(), keys)
        data = {k: to_array(self._data[k], copy=True)[idx::size] for k in keys}
        return self.__class__(data)

    def to_root(self, root: zarr.hierarchy.Group, items=None, keys=None):
        clean_group(root)
        cow = self.cow
        self._root = root
        self.cow = False
        keys = _get_keys(self.keys(), keys)
        if items is None:
            items = slice(0, len(self))
        for key in keys:
            self[key] = to_array(self[key])[items]
        self.cow = cow

    def merge(self, other, strict=True):
        if len(self) == 0:
            for k in other:
                self[k] = other[k]
            self.assert_strictness()
        else:
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

            self.assert_strictness()

    @classmethod
    def from_h5(cls, group: h5py.Group, root=None, keys=None,
            **kwargs):

        if root is not None:
            clean_group(root)

        keys = _get_keys(group.keys(), keys)

        if root is not None:
            data = root
        else:
            data = {}
        instance = cls(data, **kwargs)

        for k in keys:
            instance[k] = to_array(group[k])
        return instance

    def to_h5(self, group: h5py.Group, keys=None):
        clean_group(group)
        keys = _get_keys(self.keys(), keys)
        for k in keys:
            group.create_dataset(k, data=to_array(self[k]))

    def sample(self, idx, keys=None, root=None, **group_kwargs):
        keys = _get_keys(self.keys(), keys)
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        if root is not None:
            instance = self.__class__(root, **group_kwargs)
            for k in keys:
                instance[k] = self[k, idx]
        else:
            instance = self.__class__({k: self[k, idx] for k in keys}, **group_kwargs)
        return instance

    def random_split(self, *fractions, keys=None, seed=None, root=None, **group_kwargs):
        assert 0 < sum(fractions) <= 1
        assert all(f > 0 for f in fractions)
        idx = np.arange(len(self))
        np.random.seed(seed)
        np.random.shuffle(idx)
        sections = np.around(np.cumsum(fractions) * len(self)).astype(int)

        keys = _get_keys(self.keys(), keys)

        groups = []
        for i, sidx in enumerate(np.array_split(idx, sections)[:-1]):
            if isinstance(root, zarr.hierarchy.Group):
                data = root.create_group(f"{i:03d}")
            elif isinstance(root, list):
                data = root[i]
            else:
                data = {}
            instance = self.__class__(data, **group_kwargs)

            for k in keys:
                instance[k] = self[k, sidx]
            groups.append(instance)

        return groups

    def apply_peratom_shift(self, sap_dict, key_in='energy', key_out='energy',
            numbers_key='numbers'):
        ntyp = max(sap_dict.keys()) + 1
        sap = np.zeros(ntyp) * np.nan
        for k, v in sap_dict.items():
            sap[k] = v

        val = to_array(self._data[key_in]) - \
              sap[to_array(self._data[numbers_key])].sum(axis=-1)
        self[key_out] = val

    def iter_batched(self, batch_size=128, keys=None):
        idx = np.arange(len(self))
        idxs = np.array_split(idx, np.ceil(len(self) / batch_size))
        keys = _get_keys(self.keys(), keys)
        for idx in idxs:
            yield dict((k, to_array(v)[idx]) for k, v in self.items() if k in keys)


class Cow:
    def __init__(self, dataset, cow=True):
        self.dataset = dataset
        self.cow = cow

    def __enter__(self):
        self.dataset.flush()
        self.dataset_cow = self.dataset.cow
        self.dataset.cow = self.cow

        for group in self.dataset.values():
            group.cow = self.cow

    def __exit__(self, *args):
        self.dataset.flush()
        self.dataset.cow = self.dataset_cow

        for group in self.dataset.values():
            group.cow = self.dataset_cow


class SizeGroupedDataset:
    def __init__(self,
            data: Union[str,
                        Dict[Union[str, int], Any],
                        zarr.hierarchy.Group] = None,  # join groups as it is

            strict: bool = True,
            to_memory: bool = False,
            cow: bool = True,
            keys: Iterable = None):

        self._root = None
        self.cow = cow
        self.strict = strict
        self._data = dict()
        self._meta = dict()
        self.load_data(data, keys=keys)
        self.loader_mode = False
        self.x = {}
        self.y = {}

        if to_memory or self._root is None:
            self.to_memory(keys=keys)

    def assert_strictness(self):
        keys = set(self.datakeys())

        for g in self.values():
            assert set(g.keys()) == keys

    def load_data(self, data, keys=None):
        if isinstance(data, str) and os.path.isdir(data):
            data = zarr.convenience.open(data)

        elif isinstance(data, str) and data.endswith('.h5'):
            return self.load_h5(data, keys=keys)
        elif data is None:
            data = {}

        if isinstance(data, zarr.hierarchy.Group) or isinstance(data, dict):
            for k in data.keys():
                self[int(k)] = DataGroup(data[k], cow=self.cow, strict=self.strict, keys=keys)
        else:
            raise NotImplementedError(f"Data type {type(data)} is not supported.")

        if isinstance(data, zarr.hierarchy.Group):
            self._root = data

    def load_h5(self, data, keys=None):
        with h5py.File(data, 'r') as f:
            with Cow(self, False):
                for k, g in f.items():
                    k = int(k)
                    self[k] = DataGroup.from_h5(g, keys=keys, cow=self.cow, strict=self.strict)
            self._meta = dict(f.attrs)

    def extend_from_iterable(self, data: Iterable,
            size_fn=lambda x: len(x['numbers']),
            transform_fn=None, buffer_size=64 * 1024):
        ds = defaultdict(lambda: defaultdict(list))

        buffer = 0

        def aplly_nested_dict(d, fn):
            for k, v in d.items():
                if isinstance(v, dict):
                    d[k] = aplly_nested_dict(v, fn)
                else:
                    d[k] = fn(v)
            return d

        for d in data:
            if transform_fn is not None:
                d = transform_fn(d)
            n = size_fn(d)
            for k, v in d.items():
                ds[n][k].append(v)
            buffer += 1

            if buffer == buffer_size:
                ds = aplly_nested_dict(ds, np.stack)
                if len(self) == 0:
                    self.load_data(ds)
                else:
                    self.merge(ds)
                buffer = 0
                ds = defaultdict(lambda: defaultdict(list))
        if buffer > 0:
            ds = aplly_nested_dict(ds, np.array)
            if len(self) == 0:
                self.load_data(ds)
            else:
                self.merge(ds)

    @classmethod
    def from_iterable(cls, data: Iterable,
            root: zarr.Group = None,
            size_fn=lambda x: len(x['numbers']),
            transform_fn=None, buffer_size=64 * 1024,
            **kwargs):
        instance = cls(root, **kwargs)
        instance.extend_from_iterable(data, size_fn, transform_fn, buffer_size)
        return instance

    @classmethod
    def from_datadir(cls, path, root=None, keys=None, **kwargs):
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f'{path} does not exist or not a directory.')
        instance = cls(root, **kwargs)
        with Cow(instance, False):
            for f in glob(os.path.join(path, '???.npz')):
                k = int(os.path.basename(f)[:3])
                data = np.load(f)
                dct = dict(data)
                data.close()
                instance[k] = DataGroup(dct, keys=keys)
        return instance

    @classmethod
    def from_files(cls, files, root=None, keys=None, **kwargs):
        ins = cls(root, **kwargs)
        with Cow(ins, False):
            for fil in files:
                if not os.path.isfile(fil):
                    raise FileNotFoundError(f'{fil} does not exist or not a file.')
                k = int(os.path.splitext(os.path.basename(fil))[0])
                data = np.load(fil)
                ins[k] = DataGroup(data, keys=keys, **kwargs)
        return ins

    @classmethod
    def from_h5(cls, data: Union[str, h5py.File, h5py.Group], root: zarr.hierarchy.Group = None,
            **dataset_kwargs):
        if isinstance(data, str):
            with h5py.File(data, "r") as f:
                return cls.from_h5(f, root, **dataset_kwargs)

        elif isinstance(data, h5py.File) or isinstance(data, h5py.Group):
            keys = dataset_kwargs.get("keys", None)
            _data = root
            instance = cls(_data, **dataset_kwargs)
            with Cow(instance, False):
                for k in data.keys():
                    instance[int(k)] = DataGroup.from_h5(data[k], keys=keys,
                                                         **dataset_kwargs)
            return instance

    def to_h5(self, group: Union[str, h5py.File, h5py.Group], keys=None):

        if isinstance(group, str):
            with h5py.File(group, "w") as f:
                return self.to_h5(f, keys=keys)

        elif isinstance(group, h5py.File) or isinstance(group, h5py.Group):
            clean_group(group)
            for k in self.keys():
                subgroup = group.create_group(f"{k:03d}")
                self[k].to_h5(subgroup, keys=keys)

    def to_root(self, root: zarr.hierarchy.Group, keys=None, items=None):
        if isinstance(root, str) or isinstance(root, zarr.storage.Store):
            root = zarr.convenience.open(store=root)
        clean_group(root)
        self._root = root
        with Cow(self, False):
            for key, group in self.items():
                self[key] = group

    def to_memory(self, keys=None):
        for key in self.keys():
            self[key].to_memory(keys=keys)

    def get_shard(self, idx, size, keys=None):
        ins = self.__class__()

        for key in self.keys():
            ins[key] = self[key].get_shard(idx, size, keys=keys)
        return ins

    def save(self, store, keys=None, items=None, overwrite=False):
        if isinstance(store, str) or isinstance(store, zarr.storage.Store):
            store = zarr.group(store=store, overwrite=overwrite)

        if isinstance(store, zarr.hierarchy.Group):
            if overwrite:
                clean_group(store)
            else:
                assert len(store) == 0, "The provided storage is not empty"
            self.to_root(store, keys=keys, items=items)
        else:
            raise NotImplementedError

    def save_npz(self, dirname, namemap_fn=None, compress=False):
        os.makedirs(dirname, exist_ok=True)
        if namemap_fn is None:
            def namemap_fn(x): return f'{x:03d}.npz'
        for k, v in self.items():
            fname = os.path.join(dirname, namemap_fn(k))
            if compress:
                op = np.savez_compressed
            else:
                op = np.savez
            if len(v):
                op(fname, **{key: val for key, val in v.items()})

    def save_h5(self, filename):
        return self.to_h5(filename)

    def keys(self):
        return sorted(self._data.keys())

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def datakeys(self):
        return next(iter(self._data.values())).keys() if self._data else set()

    @property
    def groups(self):
        return self.values()

    def __len__(self):
        return sum(len(d) for d in self.values())

    def __setitem__(self, key: int, value: DataGroup):
        if not isinstance(key, int):
            raise ValueError(
                f'Failed to set key of type {type(key)}, expected int.')
        if not isinstance(value, DataGroup):
            raise ValueError(
                f'Failed to set item of wrong type. Expected DataGroup, got {type(value)}.')
        if self._data:
            if set(self.datakeys()) != set(value.keys()):
                raise ValueError(f'Wrong set of data keys.')
        value = deepcopy(value)
        if self.cow or self._root is None:
            value.to_memory()
        else:
            self._root.create_group(f"{key:03d}", overwrite=True)
            value.to_root(self._root[f"{key:03d}"])
        value.cow = self.cow
        value.strict = self.strict
        self._data[key] = value

    def __getitem__(self, item: Union[int, Tuple[int, Sequence]]) -> Union[Dict, Tuple[Dict, Dict]]:
        if isinstance(item, int):
            ret = self._data[item]
        else:
            grp, idx = item
            if self.loader_mode:
                ret = (dict((k, v[idx]) for k, v in self[grp].items() if k in self.x),
                       dict((k, v[idx]) for k, v in self[grp].items() if k in self.y))
            else:
                ret = dict((k, v[idx]) for k, v in self[grp].items())
        return ret

    def __contains__(self, value):
        return value in self.keys()

    def rename_datakey(self, old, new):
        for g in self.groups:
            g.rename_key(old, new)

    def apply(self, fn):
        for grp in self.groups:
            fn(grp)

    def flush(self, keys=None):
        if self._root is not None:
            for key, group in self.items():
                group_name = f"{key:03d}"
                if group_name not in self._root.group_keys():
                    self._root.create_group(group_name)
                group._root = self._root[group_name]
                group.flush(keys=keys)
            for key in self._root.keys():
                if int(key) not in self.keys():
                    del self._root[key]

    def merge(self, other, strict=True):
        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        if strict:
            assert set(other.datakeys()) == set(self.datakeys())
        else:
            overlaped_keys = set(other.datakeys()) & set(self.datakeys())
            for k in list(self.datakeys()):
                if k not in overlaped_keys:
                    for g in self.groups:
                        g.pop(k)

        for k in other.keys():
            if k in self:
                self[k].merge(other[k], strict=strict)
            else:
                self[k] = other[k]
                if len(self) > 0 and strict:
                    self[k].to_memory(keys=self.datakeys())
                else:
                    self[k].to_memory()

    def random_split(self, *fractions, keys=None, root=None, seed=None, **kwargs):

        clean_group(root)
        datasets = list()
        for i in range(len(fractions)):
            if isinstance(root, zarr.hierarchy.Group):
                data = root.create_group(f"{i:03d}")
            elif isinstance(root, list):
                data = root[i]
            else:
                data = {}
            datasets.append(self.__class__(data, **kwargs))

        for key, grp in self.items():
            if datasets[0]._root is not None:
                roots = [d._root.create_group(f"{key:03d}") for d in datasets]
            else:
                roots = None
            fraction_groups = grp.random_split(*fractions, keys=keys, root=roots, seed=seed,
                                               **kwargs)
            for i, fraction_group in enumerate(fraction_groups):
                datasets[i][key] = fraction_group
        return datasets

    def merge_groups(self, min_size=1, mode_atoms=False, atom_key='numbers'):
        # create list of supergroups
        sgroups = list()
        n = 0
        sg = list()
        for k, v in self.items():
            _n = len(v)
            if mode_atoms:
                _n *= v[atom_key].shape[1]
            n += _n
            sg.append(k)
            if n >= min_size:
                sgroups.append(sg)
                n = 0
                sg = list()
        sgroups[-1].extend(sg)

        # merge
        keys = self.datakeys()
        for sg in sgroups:
            for k in keys:
                arrs = [self[N][k] for N in sg]
                arrs = self._collate(arrs)
                self[sg[-1]]._data[k] = arrs
            for N in sg[:-1]:
                del self._data[N]

    @staticmethod
    def _collate(arrs, pad_value=0):
        N = sum(a.shape[0] for a in arrs)
        shape = np.stack([a.shape[1:] for a in arrs], axis=0).max(axis=0)
        arr = np.full((N, *shape), pad_value, dtype=arrs[0].dtype)
        i = 0
        for a in arrs:
            n = a.shape[0]
            slices = tuple([slice(i, i + n)] + [slice(0, x)
                                                for x in a.shape[1:]])
            arr[slices] = a
            i += n
        return arr

    def concatenate(self, key):
        try:
            C = np.concatenate([to_array(g[key]) for g in self.values()], axis=0)
        except:
            C = np.concatenate([to_array(g[key]).flatten()
                                for g in self.values()], axis=0)
        return C

    def apply_peratom_shift(self, key_in='energy', key_out='energy',
            numbers_key='numbers', sap_dict=None):
        if sap_dict is None:
            E = self.concatenate(key_in)
            ntyp = max(to_array(g[numbers_key]).max() for g in self.groups) + 1
            eye = np.eye(ntyp, dtype=np.min_scalar_type(ntyp))
            F = np.concatenate([eye[to_array(g[numbers_key])].sum(-2)
                                for g in self.values()])
            sap = np.linalg.lstsq(F, E, rcond=None)[0]
            present_elements = np.nonzero(F.sum(0))[0]
        else:
            ntyp = max(sap_dict.keys()) + 1
            sap = np.zeros(ntyp) * np.nan
            for k, v in sap_dict.items():
                sap[k] = v
            present_elements = sap_dict.keys()

        def fn(g):
            g[key_out] = to_array(g[key_in]) - sap[to_array(g[numbers_key])].sum(axis=-1)

        self.apply(fn)

        return dict((i, sap[i]) for i in present_elements)

    def apply_pertype_logratio(self, key_in='volumes', key_out='volumes',
            numbers_key='numbers', sap_dict=None):
        if sap_dict is None:
            numbers = self.concatenate('numbers')
            present_elements = sorted(np.unique(numbers))
            x = self.concatenate(key_in)
            sap_dict = dict()
            for n in present_elements:
                sap_dict[n] = np.median(x[numbers == n])
        sap = np.zeros(max(sap_dict.keys()) + 1)
        for n, v in sap_dict.items():
            sap[n] = v

        def fn(g):
            g[key_out] = np.log(to_array(g[key_in]) / sap[to_array(g[numbers_key])])

        self.apply(fn)
        return sap_dict

    def numpy_batches(self, batch_size=128, keys=None):
        for g in self.values():
            yield from g.iter_batched(batch_size, keys)

    def get_loader(self, x, y=None, batch_size=32, shuffle=True,
            num_workers=0, pin_memory=False, rank=0, world_size=1, seed=42):
        group_sizes = dict((k, len(v)) for k, v in self.items())
        batch_sizes = dict((k, batch_size) for k, v in self.items())

        self.loader_mode = True
        self.x = x
        self.y = y or {}
        sampler = SizeGrouppedSampler(
            group_sizes, batch_sizes=batch_sizes, shuffle=shuffle, rank=rank, world_size=world_size,
            seed=seed)
        loader = DataLoader(self, batch_sampler=sampler,
                            num_workers=num_workers, pin_memory=pin_memory)

        loader.collate_fn = collate_fn
        return loader

    def weighted_loader(self, x, y=None, batch_size=32, num_batches=None, num_workers=0,
            pin_memory=False, seed=42):
        num_batches = num_batches or int(len(self) / batch_size)
        self.loader_mode = True
        self.x = x
        self.y = y or {}

        if 'sample_idx' not in x:
            x.append('sample_idx')

        sampler = RandomWeightedSampler(ds=self, batch_size=batch_size, num_batches=num_batches,
                                        seed=seed)
        loader = DataLoader(self, batch_sampler=sampler,
                            num_workers=num_workers, pin_memory=pin_memory)

        loader.collate_fn = collate_fn
        return loader


class SizeGrouppedSampler:
    def __init__(self, group_sizes, batch_sizes, shuffle=False, rank=0, world_size=1, seed=None):
        self.group_sizes = group_sizes
        self.batch_sizes = batch_sizes
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self._size = sum(self._get_num_batches_for_group(N)
                         for N in self.group_sizes)
        self.epoch = 0
        self.seed = seed

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter(self._samples_list())

    def _get_num_batches_for_group(self, N):
        return int(np.ceil(self.group_sizes[N] / self.batch_sizes[N] / self.world_size))

    def _samples_list(self):
        self.epoch += 1
        seed = self.epoch + self.seed
        samples = list()
        for N, n in self.group_sizes.items():
            if n == 0:
                continue
            idx = np.arange(n)
            if self.shuffle:
                np.random.seed(seed + 1234)
                np.random.shuffle(idx)
            idx = idx[self.rank:n:self.world_size]
            n_batches = self._get_num_batches_for_group(N)
            samples.extend(((N, i),) for i in np.array_split(idx, n_batches))
        if self.shuffle:
            np.random.seed(seed + 4321)
            np.random.shuffle(samples)
        return samples


class RandomWeightedSampler:
    def __init__(self, ds, batch_size, num_batches, seed=None, uniform=False,
            peratom_batches=False):
        self.ds = ds
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.peratom_batches = peratom_batches
        self.epoch = 0
        self.seed = seed or np.random.randint(1000000)
        self.uniform = uniform
        if 'sample_weight' not in ds.datakeys():
            for g in ds.groups:
                g['sample_weight'] = np.full(len(g), 1000.0)
        if 'sample_idx' not in ds.datakeys():
            for g in ds.groups:
                g['sample_idx'] = np.arange(len(g))
        for g in ds.groups:
            g['sample_weight_upd'] = np.full(len(g), 0.0)
            g['sample_weight_upd_mask'] = np.full(len(g), False)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self._count = 0
        self.epoch += 1
        seed = self.epoch + self.seed
        np.random.seed(seed)
        if self.uniform:
            self._prepare_weights_uniform()
        else:
            self._prepare_weights()
        return self

    def __next__(self):
        if self._count == self.num_batches:
            raise StopIteration

        # g = np.random.choice(self._groups, replace=False, p=self._group_weights)
        selection = np.random.choice(np.arange(len(self._groups)), replace=False,
                                     p=self._group_weights)
        g = self._groups[selection]

        #  oops. assume ds has coord
        _n = g['coord'].shape[1]
        if self.peratom_batches:
            size = min(len(g), self.batch_size / _n)
        else:
            size = min(len(g), self.batch_size)
        idx = np.random.choice(len(g), size=min(len(g), self.batch_size), replace=False, p=g['_p'])
        self._count += 1
        return ((_n, idx),)

    def _prepare_weights_uniform(self):
        self._groups = self.ds.groups
        self._group_weights = np.array([len(g) for g in self._groups])
        self._group_weights /= self._group_weights.sum()
        for g in self._groups:
            g['_p'] = 1 / len(g)

    def _prepare_weights(self):
        self._groups = self.ds.groups
        # update sample weights
        if 'sample_weight_upd' in self.ds.datakeys():
            _u = np.concatenate([g['sample_weight_upd'] for g in self._groups])
            w = np.concatenate([g['sample_weight_upd_mask'] for g in self._groups])
            if w.any():
                _mean_weight = np.concatenate([g['sample_weight_upd'] for g in self._groups])[
                    w].mean()
                _a = 0.005 * _mean_weight
                for g in self._groups:
                    _u = g['sample_weight_upd']
                    w = g['sample_weight_upd_mask']
                    if w.any():
                        _u = _u[w]
                        _u *= np.exp(- (_u / _mean_weight) ** 2 / 20)
                        g['sample_weight'][w] = _u
                    g['sample_weight'][~w] += _a

            for g in self._groups:
                g['sample_weight_upd_mask'] = np.full(len(g), False)

                # get group weights
        self._group_weights = np.array([g['sample_weight'].sum() for g in self._groups])
        self._group_weights /= self._group_weights.sum()
        for g in self._groups:
            _w = g['sample_weight']
            g['_p'] = _w / _w.sum()
