import os
from importlib import import_module
from typing import Callable, Dict, List, Optional, Union
import yaml
from jinja2 import Template


def get_module(name: str):
    """
    Retrieves a module and function based on the given name.

    Args:
        name (str): The name of the module and function in the format 'module.function'.

    Returns:
        function: The function object.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the function does not exist in the module.
    """
    parts = name.split('.')
    mod, func = '.'.join(parts[:-1]), parts[-1]
    mod = import_module(mod)
    func = getattr(mod, func)
    return func


def get_init_module(name: str, args: List = [], kwargs: Dict = {}):
    """
    Get the initialized module based on the given name, arguments, and keyword arguments.

    Args:
        name (str): The name of the module.
        args (List, optional): The arguments to pass to the module constructor. Defaults to an empty list.
        kwargs (Dict, optional): The keyword arguments to pass to the module constructor. Defaults to an empty dictionary.

    Returns:
        The initialized module.

    """
    return get_module(name)(*args, **kwargs)


def load_yaml(config: Union[str, List, Dict], hyperpar: Optional[Union[Dict, str, None]] = None) -> Union[List, Dict]:
    """
    Load a YAML configuration file and apply optional hyperparameters.

    Args:
        config (Union[str, List, Dict]): The YAML configuration file path or a YAML object.
        hyperpar (Optional[Union[Dict, str, None]]): Optional hyperparameters to apply to the configuration.

    Returns:
        Union[List, Dict]: The loaded and processed configuration.

    Raises:
        FileNotFoundError: If a file specified in the configuration does not exist.

    """
    basedir = ''
    if isinstance(hyperpar, str):
        hyperpar = load_yaml(hyperpar)
    if isinstance(config, (list, dict)):
        if hyperpar:
            for d, k, v in _iter_rec_bottomup(config):
                if isinstance(v, str) and '{{' in v:
                    d[k] = Template(v).render(**hyperpar)
    else:
        if hasattr(config, 'read'):
            config = config.read()
        else:
            basedir = os.path.dirname(config)
            config = open(config).read()
        if hyperpar:
            config = Template(config).render(**hyperpar)
        config = yaml.load(config, Loader=yaml.FullLoader)
    # plugin yaml configs
    for d, k, v in _iter_rec_bottomup(config):
        if isinstance(v, str) and any(v.endswith(x) for x in ('.yml', '.yaml')):
            if not os.path.isfile(v):
                v = os.path.join(basedir, v)
            d[k] = load_yaml(v, hyperpar)
    return config


def _iter_rec_bottomup(d: Union[List, Dict]):
    if isinstance(d, list):
        it = enumerate(d)
    elif isinstance(d, dict):
        it = d.items()
    else:
        raise ValueError(f'Unknown type: {type(d)}')
    for k, v in it:
        if isinstance(v, (list, dict)):
            yield from _iter_rec_bottomup(v)
        yield d, k, v


def build_module(config: Union[str, Dict, List],
                 hyperpar: Union[str, Dict, None] = None) -> Union[List, Dict, Callable]:
    """
    Build a module based on the provided configuration.
    Every (possibly nested) dictionary with a 'class' key will be replaced by an instance initialized with
    arguments and keywords provided as 'args' and 'kwargs' keys.

    Args:
        config (Union[str, Dict, List]): The configuration for building the module.
        hyperpar (Union[str, Dict, None], optional): The hyperparameters for the module. Defaults to None.

    Returns:
        Union[List, Dict, Callable]: The built module.

    Raises:
        AssertionError: If `hyperpar` is provided and is not a dictionary.

    """
    if isinstance(hyperpar, str):
        hyperpar = load_yaml(hyperpar)
    if hyperpar:
        assert isinstance(hyperpar, dict)
    config = load_yaml(config, hyperpar)
    for d, k, v in _iter_rec_bottomup(config):
        if isinstance(v, dict) and 'class' in v:
            d[k] = get_init_module(v['class'], args=v.get(
                'args', []), kwargs=v.get('kwargs', {}))
    if 'class' in config:
        config = get_init_module(
            config['class'], args=config.get('args', []), kwargs=config.get('kwargs', {}))
    return config


def dict_to_dotted(d, parent=''):
    if parent:
        parent += '.'
    for k, v in list(d.items()):
        if isinstance(v, dict) and v:
            v = dict_to_dotted(v, parent + k)
            d.update(v)
            d.pop(k)
        else:
            d[parent + k] = d.pop(k)
    return d


def dotted_to_dict(d):
    for k, v in list(d.items()):
        if '.' not in k:
            continue
        ks = k.split('.')
        ds = d
        for ksp in ks[:-1]:
            if not ksp in ds:
                ds[ksp] = dict()
            ds = ds[ksp]
        ds[ks[-1]] = v
        d.pop(k)
    return d
