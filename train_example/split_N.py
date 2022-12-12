import h5py
from aimnet.data import SizeGroupedDataset, DataGroup
import numpy as np
import sys

fil_in, N  = sys.argv[1:]
N = int(N)

fil_out_basename = fil_in.split('.')[0]

out_fils = [h5py.File(f'{fil_out_basename}__{i}.h5', 'w') for i in range(N)]

with h5py.File(fil_in, 'r') as f:
    for _n, g in f.items():
        for fo in out_fils:
            fo.create_group(_n)
        idx = np.arange(len(g['coord']))
        np.random.shuffle(idx)
        idxs = np.array_split(idx, N)
        for k, v in g.items():
            v = v[()]
            for i in range(N):
                out_fils[i][_n].create_dataset(k, data=v[idxs[i]])


for f in out_fils:
    f.close()

