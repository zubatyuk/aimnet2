#!/usr/bin/env python

import sys
import torch
from aimnet.modules import Forces
from ensemble import EnsembledModel

nb, fs_in, f_out, f_out_f = sys.argv[1], sys.argv[2:-2], sys.argv[-2], sys.argv[-1]

models = [torch.jit.load(f) for f in fs_in]
models_f = list()
print('>>', len(models))

nb = bool(int(nb))

for i in range(len(models)):
    model = models[i]
    model.eval()
    model = model.cpu()
    for p in model.parameters():
        p.requires_grad_(False)
    models[i] = model
    models_f.append(Forces(model))


x = ['coord', 'numbers', 'charge']
if nb:
    x.extend(['idx_j', 'nb_pad_mask', 'idx_j_coul', 'nb_pad_mask_coul', 'shifts', 'shifts_coul'])

ens_model = EnsembledModel(models, x=x, out=['energy', 'charges'], detach=True) 
ens_model = torch.jit.script(ens_model)
ens_model.save(f_out)

ens_model = EnsembledModel(models_f, x=x, out=['energy', 'charges', 'forces'], detach=True)
ens_model = torch.jit.script(ens_model)
ens_model.save(f_out_f)

