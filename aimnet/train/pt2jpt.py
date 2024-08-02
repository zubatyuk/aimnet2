import torch
from torch import nn, Tensor
from aimnet.config import build_module, load_yaml
from typing import Optional, Dict, List
import click


def set_eval(model: nn.Module) -> torch.nn.Module:
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def add_cutoff(model: nn.Module, cutoff: Optional[float] = None, cutoff_lr : Optional[float] = float('inf')) -> nn.Module:
    if cutoff is None:
        cutoff = max(v.item() for k, v in model.state_dict().items() if k.endswith('aev.rc_s'))
    model.cutoff = cutoff
    if cutoff_lr is not None:
        model.cutoff_lr = cutoff_lr
    return model


def add_sae_to_shifts(model: nn.Module, sae_file: str) -> nn.Module:
    sae = load_yaml(sae_file)
    model.outputs.atomic_shift.double() 
    for k, v in sae.items():
        model.outputs.atomic_shift.shifts.weight[k] += v
    return model


def fix_agh(state_dict: Dict[str, Tensor]) -> dict:
    state_dict['conv_q.agh'] = state_dict['conv_q.agh'].permute(2, 1, 0).contiguous()
    state_dict['conv_a.agh'] = state_dict['conv_a.agh'].permute(2, 1, 0).contiguous()
    return state_dict


def mask_not_implemented_species(model: nn.Module, species: List[int]) -> nn.Module:
    weight = model.afv.weight
    for i in range(1, weight.shape[0]):
        if i not in species:
            weight[i, :] = torch.nan
    return model

@click.command(short_help='Compile PyTorch model to TorchScript.')
@click.argument('config', type=str)#, help='Path to the model YAML configuration file.')
@click.argument('pt', type=str)#, help='Path to the input PyTorch weights file.')
@click.argument('jpt', type=str)#, help='Path to the output TorchScript file.')
@click.option('--sae', type=str, default=None, help='Path to the energy shift YAML file.')
@click.option('--species', type=str, default=None, help='Comma-separated list of parametrized atomic numbers.')
@click.option('--fix-agh', is_flag=True, help='Fix the agh weights in the PyTorch model.')
@click.option('--no-lr', is_flag=True, help='Do not add LR cutoff for model')
def jitcompile(config, pt, jpt, sae, species, fix_agh, no_lr):
    """ Build model from YAML config, load weight from PT file and write JIT-compiled JPT file.
    Plus some modifications to work with aimnet2calc.
    """
    model: nn.Module = build_module(config)
    model = set_eval(model)
    if no_lr:
        cutoff_lr = None
    else:
        cutoff_lr = float('inf')
    model = add_cutoff(model, cutoff_lr=cutoff_lr)
    sd = torch.load(pt, map_location='cpu')
    if fix_agh:
        sd = fix_agh(sd)
    print(model.load_state_dict(sd, strict=False))
    if sae:
        model = add_sae_to_shifts(model, sae)
    numbers = None
    if species:
        numbers = list(map(int, species.split(',')))
    elif sae:
        numbers = list(load_yaml(sae).keys())
    if numbers:
        model = mask_not_implemented_species(model, numbers)
        model.register_buffer('impemented_species', torch.tensor(numbers, dtype=torch.int64))
    model_jit = torch.jit.script(model)
    model_jit.save(jpt)


if __name__ == '__main__':
    jitcompile()
