#!/usr/bin/env python
from argparse import ArgumentParser
from typing import Dict, List, Union, Callable, Optional, Sequence, Any, Tuple
import logging
import wandb
from aimnet.config import load_yaml, build_module, get_module
import os
import torch
from torch import jit, nn, optim, Tensor
import ignite.distributed as idist
from ignite.handlers import ModelCheckpoint, global_step_from_engine, TerminateOnNan
from ignite.handlers import param_scheduler
from ignite.engine import Events, _prepare_batch, Engine
from ignite.contrib.handlers.wandb_logger import WandBLogger, OptimizerParamsHandler
from ignite.contrib.handlers.tqdm_logger import ProgressBar
import re
import yaml
from aimnet.modules import Forces
import numpy as np


if os.environ.get('CUDA_DISABLE_TF32'):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', '1'))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', '0'))


if LOCAL_RANK == 0:
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.ERROR)


def make_seed():
    return int.from_bytes(os.urandom(2), 'big')


def pop_bad_ids(ds):
    if '_id' in ds.datakeys() and os.environ.get('BAD_IDS'):
        bad_ids = np.loadtxt(os.environ.get('BAD_IDS'), dtype='S')
        for _n, g in list(ds.items()):
            w = np.isin(g['_id'], bad_ids)
            if w.any():
                g = g.sample(~w)
            if len(g):
                ds._data[_n] = g
            else:
                ds._data.pop(_n)
    return ds


def load_dataset(config, typ='train'):
    keys = config['loader']['x'] + config['loader']['y']
    if os.environ.get('BAD_IDS'):
        keys.append('_id')
    ds_mod = get_module(config['class'])
    ds = ds_mod(config[typ], keys=keys, **config.get('kwargs', {}))
    logging.info(f"Loaded {typ} dataset from {config[typ]} with {len(ds)} samples.")
    # try bad ids
    _n0 = len(ds)
    ds = pop_bad_ids(ds)
    if len(ds) != _n0:
        logging.info(f"Using {len(ds)} samples for {typ}")
    # self atomic energies
    if 'energy' in keys:
        # use 'train.energy_sae' key from train config as SAE file
        # default to train dataset name with '.h5' extension replaced to '.sae'
        sae_name = config.get('energy_sae', config['train'].replace('.h5', '.sae'))
        sae = yaml.load(open(sae_name).read(), Loader=yaml.SafeLoader)
        ds.apply_peratom_shift('energy', 'energy', sap_dict=sae)
    # average volumes
    if 'volumes' in keys:
        sae_name = config['train'].replace('.h5', '.avg_vols')
        sae = yaml.load(open(sae_name).read(), Loader=yaml.SafeLoader)
        ds.apply_pertype_logratio('volumes', 'volumes')
    if 'dftd3_c6' in keys:
        sae_name = config['train'].replace('.h5', '.avg_c6')
        sae = yaml.load(open(sae_name).read(), Loader=yaml.SafeLoader)
        ds.apply_pertype_logratio('dftd3_c6', 'dftd3_c6')
    if 'dftd4_c6' in keys:
        sae_name = config['train'].replace('.h5', '.avg_c6')
        sae = yaml.load(open(sae_name).read(), Loader=yaml.SafeLoader)
        ds.apply_pertype_logratio('dftd4_c6', 'dftd4_c6')        
    if 'dftd4_alpha' in keys:
        sae_name = config['train'].replace('.h5', '.avg_alpha')
        sae = yaml.load(open(sae_name).read(), Loader=yaml.SafeLoader)
        ds.apply_pertype_logratio('dftd4_alpha', 'dftd4_alpha')
    return ds


def get_loaders(config: Dict):
    # create seed
    seed = make_seed()
    if WORLD_SIZE > 1:
        seed = idist.all_reduce(seed)

    ds_train = load_dataset(config, 'train')
    if config.get('val'):
        ds_val = load_dataset(config, 'val')
    else:
        logging.info(f'Will use 10% of train dataset for validation.')
        ds_val = ds_train.random_split(0.1)[0]
    
    # merge-pad groups
    ds_train.merge_groups(8 * config['loader']['batch_size'])
    
    # train loader
    num_batches = 500
    loader_kwargs = config['loader']
    loader_train = ds_train.weighted_loader(num_batches=num_batches, seed=seed+LOCAL_RANK, pin_memory=True, **loader_kwargs)

    # val loader
    config['loader']['batch_size'] *= 2
    loader_val = ds_val.get_loader(shuffle=False, pin_memory=True, rank=LOCAL_RANK, world_size=WORLD_SIZE, **config['loader'])

    return loader_train, loader_val


def create_supervised_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
) -> Engine:
    """Factory function for creating a trainer for supervised models.
    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
            Device can be CPU, GPU or TPU.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.
        deterministic (bool, optional): if True, returns deterministic engine of type
            :class:`~ignite.engine.deterministic.DeterministicEngine`, otherwise :class:`~ignite.engine.engine.Engine`
            (default: False).
    Note:
        `engine.state.output` for this engine is defined by `output_transform` parameter and is the loss
        of the processed batch by default.
    .. warning::
        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.
        For more information see:
        - `PyTorch Documentation <https://pytorch.org/docs/stable/optim.html#constructing-it>`_
        - `PyTorch's Explanation <https://github.com/pytorch/pytorch/issues/7844#issuecomment-503713840>`_
    Returns:
        Engine: a trainer engine with supervised update function.
    """
    def _update(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

        y_pred = model(x)

        persample_loss = loss_fn(y_pred, y)
        loss = persample_loss.mean()
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 0.4)
        optimizer.step()

        persample_loss = persample_loss.detach()
        _n = x['coord'].shape[1]
        idx = x['sample_idx'].cpu().numpy()
        engine.state.dataloader.dataset[_n]['sample_weight_upd'][idx] = persample_loss.cpu().numpy()
        engine.state.dataloader.dataset[_n]['sample_weight_upd_mask'][idx] = True

        return loss.item()

    trainer = Engine(_update)

    return trainer



def create_supervised_evaluator(
    model: torch.nn.Module,
    metrics = None,
    device = None,
    non_blocking = True,
    prepare_batch = _prepare_batch,
    output_transform = lambda x, y, y_pred: (y_pred, y),
) -> Engine:
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Note:
        `engine.state.output` for this engine is defind by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    .. warning::

        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.

        For more information see:

        - `PyTorch Documentation <https://pytorch.org/docs/stable/optim.html#constructing-it>`_

        - `PyTorch's Explanation <https://github.com/pytorch/pytorch/issues/7844#issuecomment-503713840>`_

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    metrics = metrics or {}

    def _inference(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        if not next(iter(batch[0].values())).numel():
            return None, None

        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

        with torch.no_grad():
           y_pred = model(x)

        return output_transform(x, y, y_pred)

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def wandb_init(config):
    if LOCAL_RANK == 0:
        wandb.init(config=config)
        # config = wandb.config
    if WORLD_SIZE > 1:
        config_str = yaml.dump(config)
        config_str = idist.broadcast(config_str, src=0)
        config = yaml.load(config_str, Loader=yaml.SafeLoader)
    return config


class EpochLRLogger(OptimizerParamsHandler):
    def __call__(self, engine, logger, event_name):
        global_step = engine.state.iteration
        params = {
            '{}_{}'.format(self.param_name, i): float(g[self.param_name])
            for i, g in enumerate(self.optimizer.param_groups)
        }
        logging.info(f'LR = {self.optimizer.param_groups[0]["lr"]}')
        logger.log(params, step=global_step, sync=self.sync)


def build_config(model_cfg, train_cfg, hp_cfg, hp_search_mode=False):
    # hyperparameters must be defined in `hp.yml` file
    if os.path.isfile(hp_cfg):
        hp = load_yaml(hp_cfg)
    else:
        hp = None
    # potentially modified in hp-search mode
    #if hp_search_mode:
    hp = wandb_init(hp)
    # model must be defined in `model.yml` file
    model_cfg = load_yaml(model_cfg, hp)
    # tran config must be defined in `train.yml`
    train_cfg = load_yaml(train_cfg, hp)
    if LOCAL_RANK == 0:
        with open(wandb.run.dir + '/model.yml', 'w') as f:
            f.write(yaml.dump(model_cfg))
        with open(wandb.run.dir + '/train.yml', 'w') as f:
            f.write(yaml.dump(train_cfg))

    # datasets must be defined  with AIMNET_TRAIN and AIMNET_VAL env vars
    train_cfg['data']['train'] = os.environ['AIMNET_TRAIN'].replace('__N.', '__' + str(LOCAL_RANK)+'.')
    train_cfg['data']['val'] = os.environ.get('AIMNET_VAL')
    return model_cfg, train_cfg


def build_model(config, compile=True, force_mod=None):
    model = build_module(config)
    if compile:
        model = jit.script(model)
    if force_mod is not None:
        model = force_mod(model)
    model = idist.auto_model(model, find_unused_parameters=True)
    logging.info('Build model:')
    logging.info(str(model))
    return model


def get_parameters(model: nn.Module, config: Dict) -> List:
    force_train = config.get('force_train', [])
    force_notrain = config.get('force_notrain', [])
    params = list()
    _n = 0
    logging.info(f"Trainable parameters:")
    for n, p in model.named_parameters():
        if any(re.search(x, n) for x in force_notrain):
            p.requires_grad_(False)
        if any(re.search(x, n) for x in force_train):
            p.requires_grad_(True)
        if p.requires_grad:
            params.append(p)
            logging.info(f"{n} {p.shape}")
            _n += p.numel()
    logging.info(f'Total: {_n}')
    return params


def build_optimzer(model: nn.Module, config: Dict) -> optim.Optimizer:
    optimizer_cls = get_module(config['class'])
    params = get_parameters(model, config.get('parameters', {}))
    optimizer = optimizer_cls(params, **config['kwargs'])
    return optimizer


class StopOpLowLR:
    def __init__(self, optimizer, low_lr=1e-5):
        self.low_lr = low_lr
        self.optimizer = optimizer
    def __call__(self, engine):
        if self.optimizer.param_groups[0]['lr'] < self.low_lr:
            engine.terminate()


def build_scheduler(optimizer, config: Dict) -> param_scheduler.ParamScheduler:
    if not isinstance(config, (list, tuple)):
        config = [config]
    else:
        config, durations = config[:-1], config[-1]
        assert isinstance(durations, (list, tuple))
    sched = []
    for cfg in config:
        cls = get_module(cfg['class'])
        sched.append(cls(optimizer, 'lr', **cfg.get('kwargs', {})))
        #if hasattr(sched[-1], 'event_index'):
        #    sched[-1].event_index = 1
    if len(sched) > 1:
        scheduler = param_scheduler.ConcatScheduler(schedulers=sched, durations=durations)
    else:
        scheduler = sched[0]
    return scheduler


def build_engine(net, config, loader_val=None):
    optimizer = build_optimzer(net, config['optimizer'])

    device = next(net.parameters()).device
    loss_fn = build_module(train_cfg['loss'])
    if isinstance(loss_fn, nn.Module):
        loss_fn = loss_fn.to(device)
        optimizer.param_groups[0]['params'].extend(filter(lambda p: p.requires_grad, loss_fn.parameters()))

    optimizer = idist.auto_optim(optimizer)

    metric = build_module(train_cfg['metric'])
    metric.attach_loss(loss_fn)
    trainer = create_supervised_trainer(net, optimizer, loss_fn, device=device)
    if LOCAL_RANK == 0:
        wandb_logger = WandBLogger(init=False)
        wandb_logger.attach_output_handler(trainer,
                event_name=Events.ITERATION_COMPLETED(every=200),
                output_transform=lambda loss: {"loss": loss},
                tag='train'
            )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())
    if loader_val is not None:
        validator = create_supervised_evaluator(net, metrics={'multi': metric}, device=device)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=10), validator.run, data=loader_val)
        if LOCAL_RANK == 0:
            wandb_logger.attach_output_handler(
                    validator,
                    event_name=Events.EPOCH_COMPLETED,
                    global_step_transform=lambda *_: trainer.state.iteration,
                    metric_names="all",
                    tag='val',
                )

    if 'rop_scheduler' in config:
        terminator = StopOpLowLR(optimizer, config['rop_scheduler'].pop('low_lr', 1e-5))
        trainer.add_event_handler(Events.EPOCH_STARTED, terminator)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['rop_scheduler'])
        @validator.on(Events.COMPLETED)
        def reduct_step(engine):
            scheduler.step(engine.state.metrics['loss'])
    elif 'scheduler' in config:
        scheduler = build_scheduler(optimizer, config['scheduler'])
        trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

    if LOCAL_RANK == 0:
        wandb_logger.attach(
            trainer,
            log_handler=EpochLRLogger(optimizer),
            event_name=Events.EPOCH_STARTED
            )

        if loader_val is not None:
            chk_engine = validator
            score_function = lambda engine: 1.0 / engine.state.metrics['loss']
        else:
            chk_engine = trainer
            score_function = lambda engine: engine.state.iteration

        model_checkpoint = ModelCheckpoint(
            wandb_logger.run.dir, n_saved=1, filename_prefix='best',
            require_empty=False, score_function=score_function,
            global_step_transform=global_step_from_engine(trainer)
        )
        net = unwrap_module(net)
        chk_engine.add_event_handler(Events.EPOCH_COMPLETED, model_checkpoint, {'model': net})

    return trainer


def unwrap_module(net):
    if isinstance(net, Forces):
        net = net.module
    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
        net = net.module
    return net


def run(local_rank, model_cfg, train_cfg, load, save):
    model = build_model(model_cfg, compile=False)
    if load is not None:
        device = next(model.parameters()).device
        logging.info(f'Loading weights from file {load}')
        sd = torch.load(load, map_location=device)
        logging.info(unwrap_module(model).load_state_dict(sd, strict=False))
    train_loader, val_loader = get_loaders(train_cfg['data'])
    if 'forces' in next(iter(train_loader))[1]:
        model = Forces(model)
    trainer = build_engine(model, train_cfg, val_loader)
    if local_rank == 0:
        pbar = ProgressBar()
        pbar.attach(trainer, metric_names='all', event_name=Events.ITERATION_COMPLETED(every=200))
    max_epochs = train_cfg.get('epochs', 100)
    trainer.run(train_loader, max_epochs=max_epochs)
    if save is not None:
        torch.save(unwrap_module(model).state_dict(), save)


if __name__ == '__main__':
    import argparse

    parser = ArgumentParser()
    parser.add_argument('--model_def', type=str, default='model.yml')
    parser.add_argument('--train_def', type=str, default='train.yml')
    parser.add_argument('--hp_def', type=str, default='hp.yml')
    parser.add_argument('--hp_search_mode', action='store_true')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()


    num_gpus = torch.cuda.device_count()
    logging.info(f'Start training using {num_gpus} GPU(s):')
    for i in range(num_gpus):
        logging.info(torch.cuda.get_device_name(i))

    model_cfg, train_cfg = build_config(args.model_def, args.train_def, args.hp_def, args.hp_search_mode)
    logging.info(yaml.dump(model_cfg))
    logging.info(yaml.dump(train_cfg))

    if num_gpus > 1:
        with idist.Parallel(backend='nccl') as parallel:
            parallel.run(run, model_cfg, train_cfg, args.load, args.save)
    else:
        run(0, model_cfg, train_cfg, args.load, args.save)

