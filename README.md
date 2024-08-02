# AIMNet2

Models are compatible for use with [aimnet2calc](https://github.com/isayevlab/AIMNet2)

  
## Installation

Create conda environment
```
conda create -n aimnet2 python=3.11
conda activate aimnet
```  

Install PyTorch with a proper CUDA version according to instructions at [pytorch.org](https://pytorch.org). E.g.
```
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
```

Install other dependencies.
```
conda install -c conda-forge -c pytorch -c nvidia -f requirements.txt
```

Finally, install using setuptools.
```
python setup.py install
```

## Training examples

Quick-start training examples provided in [train.md](https://github.com/zubatyuk/aimnet2/blob/master/aimnet/train/train.md)
