import torch
import sys
from aimnet.config import build_module, load_yaml

config, sae, pt, jpt = sys.argv[1:]


model = build_module(config)
for p in model.parameters():
    p.requires_grad_(False)
model.eval()
model = torch.jit.script(model)
d = torch.load(pt, map_location='cpu')
#for k, v in list(d.items()):
#    if k.startswith('mod.'):
#        d[k[4:]] = d.pop(k)
sae = load_yaml(sae)
d['outputs.1.shifts.weight'] = d['outputs.1.shifts.weight'].to(torch.double)
for k, v in sae.items():
    d['outputs.1.shifts.weight'][k] += v
print(model.load_state_dict(d, strict=False))
model.save(jpt)
