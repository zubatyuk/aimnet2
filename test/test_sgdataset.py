import numpy as np
import zarr

from aimnet.data import SizeGroupedDataset
from aimnet.data.sgdataset import clean_group


def test_basic_functionality():
    dataset = SizeGroupedDataset()
    root = zarr.group("test_zarr")
    clean_group(root)
    r1 = root.create_group("dataset_1")
    r2 = root.create_group("dataset_2")

    dataset = SizeGroupedDataset.from_h5("test.h5", root=r1)
    dataset.to_root(r2)

    assert dataset._root == r2

    in_memory_dataset = SizeGroupedDataset(r1)
    in_memory_dataset.to_memory()

    length = len(dataset)
    keys = dataset.keys()
    dataset.items()
    dataset.values()
    dataset.groups
    dataset.rename_datakey("energy", "energiya")

    try:
        dataset.merge(in_memory_dataset)
        assert False
    except AssertionError:
        dataset.merge(in_memory_dataset, strict=False)

    assert "energiya" not in dataset.datakeys()

    print(len(dataset))
    assert len(dataset) == 2 * length

    key = next(iter(keys))
    assert len(dataset._root[f"{key:03d}/numbers"]) != len(dataset[key])
    dataset.flush()

    assert len(dataset._root[f"{key:03d}/numbers"]) == len(dataset[key])

    clean_group(root)


def test_split_functionality():
    root = zarr.group("test_zarr")
    clean_group(root)
    data_group = root.create_group("data")
    test_group = root.create_group("test")
    train_group = root.create_group("train")

    dataset = SizeGroupedDataset.from_h5("test.h5", root=data_group)

    inmem_test, inmem_train = dataset.random_split(0.1, 0.9)

    inmem_test, inmem_valid = dataset.random_split(0.1, 0.1)

    assert abs(len(inmem_test) / len(dataset) - 0.1) < 0.05
    assert abs(len(inmem_train) / len(dataset) - 0.9) < 0.05

    ondisk_train, ondisk_test = dataset.random_split(0.1, 0.9, root=[test_group, train_group])

    inmem_test, _ = dataset.random_split(0.1, 0.1, seed=42)
    inmem_test_copy, _ = dataset.random_split(0.1, 0.1, seed=42)

    inmem_test_diff, _ = dataset.random_split(0.1, 0.1, seed=41)

    key = next(iter(inmem_test_copy.keys()))
    assert np.mean(np.abs(inmem_test_copy[key]["energy"] - inmem_test[key]["energy"])) < 1e-6
    assert np.mean(np.abs(inmem_test_diff[key]["energy"] - inmem_test[key]["energy"])) > 1e-3

    inmem_test, = dataset.random_split(0.1, )

    assert abs(len(inmem_test) / len(dataset) - 0.1) < 0.05


def test_loaders():
    # todo: add loader tests
    pass
