import click
from aimnet.data import SizeGroupedDataset
import numpy as np
import logging


@click.command(short_help='Calculate SAE for a dataset.')
@click.option('--samples', type=int, default=100000, help='Max number of samples to consider.')
@click.argument('ds', type=str)
@click.argument('output', type=str)
def calc_sae(ds, output, samples=100000):
    """ Script to calculate energy SAE for a dataset DS. Writes SAE to OUTPUT file.
    """
    logging.info(f'Loading dataset from {ds}')
    ds = SizeGroupedDataset(ds, keys=['numbers', 'energy'])
    logging.info(f'Loaded dataset with {len(ds)} samples')
    if samples > 0 and len(ds) > samples:
        ds = ds.random_split(samples / len(ds))[0]
    logging.info(f'Using {len(ds)} samples to calculate SAE')
    sae = ds.apply_peratom_shift('energy', '_energy')
    # remove up 2 percentiles from right and left
    energy = ds.concatenate('_energy')
    pct1, pct2 = np.percentile(energy, [2, 98])
    for _n, g in ds.items():
        mask = (g['_energy'] > pct1) & (g['_energy'] < pct2)
        g = g.sample(mask)
        if not len(g):
            ds._data.pop(_n)
        else:
            ds[_n] = g
    # now re-compute SAE
    sae = ds.apply_peratom_shift('energy', '_energy')
    # save
    with open(output, 'w') as f:
        for k, v in sae.items():
            str = f'{k}: {v}\n'
            f.write(f'{k}: {v}\n')
            print(str, end='')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    calc_sae()





              
