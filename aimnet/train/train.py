import click
from omegaconf import OmegaConf
from aimnet.config import build_module
import os
import torch
from ignite import distributed as idist
from ignite.handlers.tqdm_logger import ProgressBar
import logging
from aimnet.modules import Forces
from aimnet.train import utils


_default_model = os.path.join(os.path.dirname(__file__), '..', 'models', 'aimnet2.yaml')
_default_config = os.path.join(os.path.dirname(__file__), 'default_train.yaml')

@click.command()
@click.option('--config', type=click.Path(exists=True), default=None,
    help='Path to the extra configuration file (overrides values the default config).'
    )
@click.option('--model', type=click.Path(exists=True), default=_default_model,
    help='Path to the model definition file.'
    )
@click.option('--load', type=click.Path(exists=True), default=None,
    help='Path to the model weights to load.'
    )
@click.option('--save', type=click.Path(), default=None,
    help='Path to save the model weights.'
    )
@click.argument('args', type=str, nargs=-1)
def train(config, model, load, save, args):
    """Train AIMNet2 model.
    By default, will load AIMNet2 model and default train config.
    ARGS are one or more parameters wo overwrite in config in a dot-separated form.
    For example: `train.data=mydataset.h5`.
    """
    logging.basicConfig(level=logging.INFO)

    # model config
    logging.info('Start training')
    logging.info(f'Using model definition: {model}')
    model_cfg = OmegaConf.load(model)
    logging.info('--- START model.yaml ---')
    model_cfg = OmegaConf.to_yaml(model_cfg)
    logging.info(model_cfg)
    logging.info('--- END model.yaml ---')

    # train config
    logging.info(f'Using default training configuration: {_default_config}')
    train_cfg = OmegaConf.load(_default_config)
    if config is not None:
        logging.info(f'Using additional configuration: {config}')
        train_cfg = OmegaConf.merge(train_cfg, OmegaConf.load(config))
    if args:
        logging.info(f'Overriding configuration:')
        for arg in args:
            logging.info(arg)
        args_cfg = OmegaConf.from_dotlist(args)
        train_cfg = OmegaConf.merge(train_cfg, args_cfg)
    logging.info('--- START train.yaml ---')
    train_cfg = OmegaConf.to_yaml(train_cfg)
    logging.info(train_cfg)
    logging.info('--- END train.yaml ---')

    # launch
    num_gpus = torch.cuda.device_count()
    logging.info(f'Start training using {num_gpus} GPU(s):')
    for i in range(num_gpus):
        logging.info(torch.cuda.get_device_name(i))
    if num_gpus == 0:
        logging.warning('No GPU available. Training will run on CPU. Use for testing only.')
    if num_gpus > 1:
        logging.info('Using DDP training.')
        with idist.Parallel(backend='nccl') as parallel:
            parallel.run(run, model_cfg, train_cfg, load, save)
    else:
        run(0, model_cfg, train_cfg, load, save)


def run(local_rank, model_cfg, train_cfg, load, save):
    if local_rank == 0:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    # load configs
    model_cfg = OmegaConf.create(model_cfg)
    train_cfg = OmegaConf.create(train_cfg)

    # build model
    _force_training = 'forces' in train_cfg.data.y
    model = utils.build_model(model_cfg, forces=_force_training)
    model = utils.set_trainable_parameters(model,
            train_cfg.optimizer.force_train,
            train_cfg.optimizer.force_no_train)
    model = idist.auto_model(model)

    # load weights
    if load is not None:
        device = next(model.parameters()).device
        logging.info(f'Loading weights from file {load}')
        sd = torch.load(load, map_location=device)
        logging.info(utils.unwrap_module(model).load_state_dict(sd, strict=False))

    # data loaders
    train_loader, val_loader = utils.get_loaders(train_cfg.data)

    # optimizer, scheduler, etc
    optimizer = utils.get_optimizer(model, train_cfg.optimizer)
    optimizer = idist.auto_optim(optimizer)
    if train_cfg.scheduler is not None:
        scheduler = utils.get_scheduler(optimizer, train_cfg.scheduler)
    else:
        scheduler = None
    loss = utils.get_loss(train_cfg.loss)
    metrics = utils.get_metrics(train_cfg.metrics)
    metrics.attach_loss(loss)

    # ignite engine
    trainer, validator = utils.build_engine(model, optimizer, scheduler, loss, metrics, train_cfg, val_loader)

    if local_rank == 0 and train_cfg.wandb is not None:
        utils.setup_wandb(train_cfg, model_cfg, model, trainer, validator, optimizer)
        
    trainer.run(train_loader, max_epochs=train_cfg.trainer.epochs)

    if local_rank == 0 and save is not None:
        logging.info(f'Saving model weights to file {save}')
        torch.save(utils.unwrap_module(model).state_dict(), save)

    
if __name__ == '__main__':
    train()        




        





    



    



    

    

    







