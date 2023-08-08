import time

import numpy as np
import zarr

from aimnet.data import SizeGroupedDataset
from aimnet.data.sgdataset import clean_group


def test_reding_and_writing():
    dataset = SizeGroupedDataset()

    assert len(dataset) == 0

    in_mem = SizeGroupedDataset.from_h5("test_data/test.h5")
    in_mem.save("test_data/test_zarr", overwrite=True)

    zarr_dataset = SizeGroupedDataset("test_data/test_zarr")

    try:
        zarr_dataset.save("test_data/test_zarr")
        exit()
    except AssertionError:
        pass

    zarr_dataset.save_npz("test_data/npz")
    dataset = SizeGroupedDataset.from_datadir("test_data/npz")

    assert len(zarr_dataset) == len(dataset)
    from_h5_dataset = SizeGroupedDataset("test_data/test.h5")
    assert len(zarr_dataset) == len(from_h5_dataset)

    zarr_dataset.save_h5("test_data/another_test.h5")
    h5_dataset = SizeGroupedDataset("test_data/another_test.h5")
    from_zarr_dataset = SizeGroupedDataset("test_data/test_zarr", to_memory=True)

    item = from_zarr_dataset[from_zarr_dataset.keys()[0], 0]

    iterable = [item] * 10000

    ds = SizeGroupedDataset.from_iterable(iterable, zarr.group(), buffer_size=1000)

    assert len(ds) == 10000

    l = len(dataset)
    dataset.extend_from_iterable(iterable)

    assert len(dataset) == l + 10000

    empty_dataset = SizeGroupedDataset("./")

    empty_dataset.merge(zarr_dataset, strict=False)

    ln = len(empty_dataset)

    empty_dataset.merge(zarr_dataset)

    assert ln == len(empty_dataset) // 2

    zarr_dataset.rename_datakey("numbers", "n")
    del zarr_dataset._data[27]

    empty_dataset.merge(zarr_dataset, strict=False)

    try:
        empty_dataset.merge(zarr_dataset)
        exit()
    except AssertionError:
        pass




def test_basic_functionality():
    root = zarr.group("test_data/test_zarr")
    clean_group(root)
    r1 = root.create_group("dataset_1")
    r2 = root.create_group("dataset_2")

    dataset = SizeGroupedDataset.from_h5("test_data/test.h5", root=r1)
    dataset.to_root(r2)

    dataset.apply_peratom_shift("energy", "changed_energy")
    dataset.apply_peratom_shift("energy", "another_changed_energy")
    dataset.flush(keys=["changed_energy"])

    assert isinstance(dataset[dataset.keys()[0]]["changed_energy"], zarr.Array)
    assert isinstance(dataset[dataset.keys()[0]]["another_changed_energy"], np.ndarray)

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

    shard = dataset.get_shard(0, 2)

    assert abs(len(shard) * 2 - len(dataset)) <= 1

    clean_group(root)


def test_split_functionality():
    root = zarr.group("test_data/test_zarr")
    clean_group(root)
    data_group = root.create_group("data")
    test_group = root.create_group("test")
    train_group = root.create_group("train")

    dataset = SizeGroupedDataset.from_h5("test_data/test.h5", root=data_group)

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
    dataset = SizeGroupedDataset.from_h5("test_data/test.h5")
    dataset.to_root(zarr.group())

    dataloader = dataset.get_loader(x=["coordinates", "numbers"], y=["energy"])

    n_batches = len(dataloader)
    curr_time = time.time()

    for _ in dataloader:
        pass
    on_disk_rate = n_batches / (time.time() - curr_time)

    dataset.to_memory()
    curr_time = time.time()

    for _ in dataloader:
        pass
    in_mem_rate = n_batches / (time.time() - curr_time)

    assert in_mem_rate / 20 > on_disk_rate
