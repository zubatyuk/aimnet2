import os
from glob import glob
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import zarr

from aimnet.data.sgdataset import DataGroup


# from torch.utils.data.dataloader import DataLoader, default_collate


class ZarrGroup:
    def __init__(self, group: zarr.hierarchy.Group, data: Dict[str, np.ndarray] = None,
            keys=None, overwrite=False):

        if keys is None:
            keys = set()

        if data is None:
            data = dict()

        self._group = group

        array_keys = set(data.keys())
        group_keys = set(group.array_keys())
        keys = set(keys)
        if len(keys) == 0:
            if len(group_keys) != 0:
                keys = group_keys

            elif len(array_keys) != 0:
                keys = array_keys

            else:
                raise "Failed to create an empty group"

        if overwrite or len(group_keys) == 0:
            if len(array_keys) == 0:
                raise 'Failed to overwrite a group with empty dictionary'

            if array_keys & keys != keys:
                raise f'Failed to find all of the required keys'
            _n = None
            for key in keys:
                self[key] = data[key]
                if _n is None:
                    _n = len(data[key])
                assert len(data[key]) == _n

    def __getitem__(self, key):
        return self._group[key]

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise ValueError(
                f'Failed to set key of type {type(key)}, expected str.')
        if not isinstance(value, np.ndarray):
            raise ValueError(
                f'Failed to set item of wrong type. Expected {type(np.ndarray)}, got {type(value)}.')
        if len(self) and len(value) != len(self):
            raise ValueError(
                f'Failed to set item of wrong shape. Expected {len(self)}, got {len(value)}.')
        self._group.array(key, value, overwrite=True, dtype=value.dtype)

    def __delitem__(self, key):
        del self._group[key]

    def __contains__(self, key):
        return key in self._group

    def __len__(self):
        return len(next(iter(self.values()))) if len(self.keys()) else 0

    def to_dict(self):
        return {k: v[:] for k, v in self.items()}

    def items(self):
        return [(key, value) for key, value in self._group.items() if key in self.keys()]

    def values(self):
        return self.to_dict().values()

    def keys(self):
        return sorted(list(self._group.array_keys()))

    def pop(self, key):
        item = self[key][:]
        del self[key]
        return item

    def rename_key(self, old, new):
        self[new] = self.pop(old)

    def sample(self, idx, keys=None):
        if keys is None:
            keys = self.keys()
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        return DataGroup(dict((k, self[k][idx]) for k in keys))

    def random_split(self, *fractions, seed=None):
        assert 0 < sum(fractions) <= 1
        assert all(f > 0 for f in fractions)
        idx = np.arange(len(self))
        np.random.seed(seed)
        np.random.shuffle(idx)
        sections = np.around(np.cumsum(fractions) * len(self)).astype(np.int)
        return [self.sample(sidx) if len(sidx) else DataGroup() for sidx in
                np.array_split(idx, sections)]

    def cv_split(self, cv: int = 5, seed=None):
        """ Return list of `cv` tuples containing train and val `DataGroup`s
        """
        fractions = [1 / cv] * cv
        parts = self.random_split(*fractions, seed=seed)
        splits = list()
        for icv in range(cv):
            val = parts[icv]
            _idx = [_i for _i in range(cv) if _i != icv]
            train = parts[_idx[0]]
            train.cat(*[parts[_i] for _i in _idx[1:]])
            splits.append((train, val))
        return splits

    def shuffle(self, seed=None):
        idx = np.arange(len(self))
        np.random.seed(seed)
        np.random.shuffle(idx)
        for k, v in self.items():
            self[k] = v[idx]

    def cat(self, *others):
        _n = set(self.keys())
        for other in others:
            assert set(other.keys()) == _n
        for k in self.keys():
            for other in others:
                self._group[k].append(other[k])

    def iter_batched(self, batch_size=128, keys=None):
        idx = np.arange(len(self))
        idxs = np.array_split(idx, np.ceil(len(self) / batch_size))
        if keys is None:
            keys = self.keys()
        for idx in idxs:
            yield dict((k, v[idx]) for k, v in self.items() if k in keys)

    def merge(self, other, strict=True):
        if strict:
            assert set(self.keys()) == set(other.keys())
            keys = self.keys()
        else:
            keys = set(self.keys()) & set(other.keys())
        for k in list(self.keys()):
            if k in keys:
                self._group[k].append(other[k])
            else:
                del self[k]

    def apply_peratom_shift(self, sap_dict, key_in='energy', key_out='energy',
            numbers_key='numbers'):
        ntyp = max(sap_dict.keys()) + 1
        sap = np.zeros(ntyp) * np.nan
        for k, v in sap_dict.items():
            sap[k] = v
        self[key_out] = self[key_in][:] - \
                        sap[self[numbers_key][:]].sum(axis=-1)



class SizeGroupedDataset:
    def __init__(self, storage, data: Union[
        str, List[str], Dict[int, str], Dict[int, Dict[str, np.ndarray]], None] = None, keys=None):
        self._root = zarr.group(storage)
        self._meta = dict()
        self._groups = dict()
        if isinstance(data, str):
            if os.path.isdir(data):
                self.load_datadir(data, keys=keys)
            else:
                self.load_h5(data, keys=keys)
        elif isinstance(data, (list, tuple)):
            self.load_files(data)
        elif isinstance(data, dict):
            self.load_dict(data)
        self.loader_mode = False
        self.x = {}
        self.y = {}

    def load_datadir(self, path, keys=None):
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f'{path} does not exist or not a directory.')
        for f in glob(os.path.join(path, '???.npz')):
            k = int(os.path.basename(f)[:3])
            self._root.create_group(f"{k:03}")
            npz = np.load(f)
            self[k] = ZarrGroup(group=self._root[f"{k:03}"],
                                data=npz,
                                keys=keys)

    # def load_files(self, files, keys=None):
    #     for fil in files:
    #         if not os.path.isfile(fil):
    #             raise FileNotFoundError(f'{fil} does not exist or not a file.')
    #         k = int(os.path.splitext(os.path.basename(fil))[0])
    #         self[k] = DataGroup(fil, keys=keys)

    def load_dict(self, data, keys=None):
        for k, v in data.items():
            self._root.create_group(f"{k:03}")
            self[k] = ZarrGroup(group=self._root[f"{k:03}"],
                                data=v,
                                keys=keys)

    # def load_h5(self, data, keys=None):
    #     with h5py.File(data, 'r') as f:
    #         for k, g in f.items():
    #             k = int(k)
    #             self[k] = DataGroup(g, keys=keys)
    #         self._meta = dict(f.attrs)

    def keys(self):
        return sorted([int(i) for i in self._root.group_keys()])

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def datakeys(self):
        return next(iter(self._root.values())).keys() if len(self) else set()

    @property
    def groups(self):
        return self.values()

    def __len__(self):
        return sum(len(d) for d in self.values())

    def __setitem__(self, key: int, value):
        if not isinstance(key, int):
            raise ValueError(
                f'Failed to set key of type {type(key)}, expected int.')
        if not isinstance(value, ZarrGroup):
            raise ValueError(
                f'Failed to set item of wrong type. Expected ZarrGroup, got {type(value)}.')
        if len(self):
            if set(self.datakeys()) != set(value.keys()):
                raise ValueError(f'Wrong set of data keys.')

        self._groups[key] = value

    def __getitem__(self, item: Union[int, Tuple[int, Sequence]]) -> Union[Dict, Tuple[Dict, Dict]]:
        if isinstance(item, int):
            ret = self._groups[item]
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

    @classmethod
    def from_h5(cls, h5file):
        pass

    def rename_datakey(self, old, new):
        for g in self._groups:
            g.rename_key(old, new)

    def apply(self, fn):
        for grp in self._groups:
            fn(grp)

    def merge(self, other, strict=True):
        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        if strict:
            assert set(other.datakeys()) == set(self.datakeys())
        else:
            keys = set(other.datakeys()) & set(self.datakeys())
            for k in list(self.datakeys()):
                if k not in keys:
                    for g in self.groups:
                        g.pop(k)
            for k in list(other.datakeys()):
                if k not in keys:
                    for g in other.groups:
                        g.pop(k)
        for k in other.keys():
            if k in self:
                self[k].cat(other[k])
            else:
                self[k] = other[k]

    def random_split(self, *fractions, seed=None):
        splitted_groups = dict()
        for k, v in self.items():
            splitted_groups[k] = v.random_split(*fractions, seed=seed)
        datasets = list()
        for i in range(len(fractions)):
            datasets.append(self.__class__(
                dict((k, splitted_groups[k][i]) for k in splitted_groups if
                     len(splitted_groups[k][i]) > 0)))
        return datasets

    def cv_split(self, cv: int = 5, seed=None):
        splitted_groups = dict()
        for k, v in self.items():
            splitted_groups[k] = v.cv_split(cv, seed)
        datasets = list()
        for i in range(cv):
            train = self.__class__(
                dict((k, splitted_groups[k][i][0]) for k in splitted_groups))
            val = self.__class__(
                dict((k, splitted_groups[k][i][1]) for k in splitted_groups))
            datasets.append((train, val))
        return datasets

    def shuffle(self, seed=None):
        for v in self.values():
            v.shuffle(seed)

    def save(self, dirname, namemap_fn=None, compress=False):
        os.makedirs(dirname, exist_ok=True)
        if namemap_fn is None:
            def namemap_fn(x): return f'{x:03d}.npz'
        for k, v in self.items():
            fname = os.path.join(dirname, namemap_fn(k))
            v.save(fname, compress=compress)

    def save_h5(self, filename):
        with h5py.File(filename, 'w') as f:
            for n, g in self.items():
                n = f'{n:03d}'
                h5g = f.create_group(n)
                for k, v in g.items():
                    h5g.create_dataset(k, data=v)
                for k, v in self._meta.items():
                    f.attrs[k] = v

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
            C = np.concatenate([g[key] for g in self.values()], axis=0)
        except:
            C = np.concatenate([g[key].flatten()
                                for g in self.values()], axis=0)
        return C

    def apply_peratom_shift(self, key_in='energy', key_out='energy',
            numbers_key='numbers', sap_dict=None):
        if sap_dict is None:
            E = self.concatenate(key_in)
            ntyp = max(g[numbers_key].max() for g in self.groups) + 1
            eye = np.eye(ntyp, dtype=np.min_scalar_type(ntyp))
            F = np.concatenate([eye[g[numbers_key]].sum(-2)
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
            g[key_out] = g[key_in] - sap[g[numbers_key]].sum(axis=-1)

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
            g[key_out] = np.log(g[key_in] / sap[g[numbers_key]])

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

        def _squeeze(t):
            for d in t:
                for v in d.values():
                    v.squeeze_(0)
            return t

        loader.collate_fn = lambda x: _squeeze(default_collate(x))
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

        def _squeeze(t):
            for d in t:
                for v in d.values():
                    v.squeeze_(0)
            return t

        loader.collate_fn = lambda x: _squeeze(default_collate(x))
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


if __name__ == "__main__":
    root = zarr.group()

    group = root.create_group("group")

    item = {"indices": np.random.randint(0, 10, (10, 30)),
            "forces": np.random.random((10, 30, 3))}

    zarr_group = ZarrGroup(group, item)

    zarr_group["indices"] = item["indices"]
    zarr_group["another_forces"] = item["forces"]

    another_group = root.create_group("another_group")

    item = {"indices": np.random.randint(0, 10, (10, 30)),
            "forces": np.random.random((10, 30, 3)),
            "another_forces": np.random.random((10, 30, 3))}

    another_zarr_group = ZarrGroup(another_group, item)

    zarr_group.cat(another_zarr_group)
    print(len(zarr_group))
    print(len(another_zarr_group))

    another_zarr_group["other_forces"] = np.random.random((10, 30, 3))
    try:
        zarr_group.merge(another_zarr_group)
    except Exception as e:
        print(e)

    zarr_group.merge(another_zarr_group, strict=False)
