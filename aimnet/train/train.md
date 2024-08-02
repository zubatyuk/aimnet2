# AIMNet2 training examples.

## General workflow

### 1. Prepare data.

Training dataset should be in form of HDF5 file with group containing molecules of the same size. Here is an example for a dataset containing 25768 28-atom and 19404 29-atom molecules.
```
$ h5ls -r dataset.h5
...
/028                     Group
/028/charge              Dataset {25768}
/028/charges             Dataset {25768, 28}
/028/coord               Dataset {25768, 28, 3}
/028/energy              Dataset {25768}
/028/forces              Dataset {25768, 28, 3}
/028/numbers             Dataset {25768, 28}
/029                     Group
/029/charge              Dataset {19404}
/029/charges             Dataset {19404, 29}
/029/coord               Dataset {19404, 29, 3}
/029/energy              Dataset {19404}
/029/forces              Dataset {19404, 29, 3}
/029/numbers             Dataset {19404, 29}
...
```
Units should be based on Angstrom, electron-volt and electron charge.

### 2. Prepare training config

Training script (after you installed with setuptools)
```
$ aimnet train --help
```
To actually training training, few pieces are needed:

- Training config. 
	`aimnet/train/default_train.yaml` is used as a base, which could be extended with command line options or another YAML config which will overwrite values in `aimnet/train/default_train.yaml`. At very least, `run_name` should be set.

- Model definition YAML file. By default, `aimnet/models/aimnet2.yaml` will be used.

- File with self-atomic energies. It could be prepared with
``` $ aimnet calc_sae dataset.h5 dataset_sae.yaml```

### 3. Wandb logging
Training script uses (Wandb)[https://wandb.ai/] logging. It is free for personal and academic use. To track train progress, either (wandb.ai)[wandb,ai] account is required,  or  a local Docker-based server launched with `wandb server`. By default, wandb is set to offline mode.
To setup online W&B account, use
`$ wandb login`
To set up W&B project and entity name, create extra configuration, save it to a file named, for example, `extra_conf.yaml` and pass it to `aimnet train` with `--config extra_conf.yaml`parameter.
```
wandb:
	init:
		mode: online
		entity: username
		project: project_name
```
### 4. Launch training
For good loader performance, disable multithreading in numpy
`$ export OMP_NUM_THREADS=1`

By default, training will be launched on all available GPUs in single-node, distributed data parallel manner. If training on a single GPU is desirable:
` $ export CUDA_VISIBLE_DEVICES=0`

Finally, launch training script with all default parameters, setting `run_name` in command line. 
```
$ aimnet train data.train=dataset.h5 data.sae.energy.file=dataset_sae.yaml run_name=firstrun
```
### 5. Compile trained model for use with [aimnet2calc](https://github.com/isayevlab/AIMNet2)
`$ aimnet jitcompile my_model.pt my_model.jpt`