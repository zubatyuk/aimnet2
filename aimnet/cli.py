import click
import aimnet
from aimnet.train.train import train
from aimnet.train.pt2jpt import jitcompile
from aimnet.train.calc_sae import calc_sae


@click.group()
def cli():
    """AIMNet2 command line tool
    """


cli.add_command(train, name='train')
cli.add_command(jitcompile, name='jitcompile')
cli.add_command(calc_sae, name='calc_sae')


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    cli()
