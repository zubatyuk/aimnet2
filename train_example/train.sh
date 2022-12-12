#!/bin/bash

DS_TRAIN=$1
DS_VAL=$2
MODEL=$3
export RUN_NAME=$4
export SKIP_IDS=$5

wandb on

export PYTHONPATH=${HOME}/aimnet2
export AIMNET_TRAIN=${DS_TRAIN}
export AIMNET_VAL=${DS_VAL}

echo $AIMNET_TRAIN
echo $AIMNET_VAL

export WANDB_PROJECT=pc14i_v2_fin2_v2

#export CUDA_LAUNCH_BLOCKING=1

for i in `set | grep ^SLURM | tr '=', ' ' | awk '{print $1}'`; do 
  unset $i
done

export WANDB_NAME=${RUN_NAME}
python -m torch.distributed.launch --nproc_per_node=4 --use_env train_mp.py --train_def train.yml --model_def $MODEL --save ${RUN_NAME}.pt &> ${RUN_NAME}.log

#python -m torch.distributed.launch --nproc_per_node=3 --use_env train_mp_enet.py --train_def train_enet_ranger21.yml --model_def model_enet.yml --save ${RUN_NAME}_enet.pt &> ${RUN_NAME}_enet_rng.log

#export QNET_JPT=v2b_c_qnet.jpt
#export WANDB_NAME=${RUN_NAME}_enet
#python -m torch.distributed.launch --nproc_per_node=3 --use_env train_mp_enet.py --train_def train_enet.yml --model_def model_enet.yml --save ${RUN_NAME}_enet.pt --load _model.pt &> ${RUN_NAME}_enet_nonise.log


#python -m torch.distributed.launch --nproc_per_node=3 --use_env train_mp.py --train_def train.yml --model_def model.yml &> ${RUN_NAME}.log
#python -m torch.distributed.launch --nproc_per_node=3 --use_env train_mp_enet.py --train_def train_enet.yml --model_def model_enet.yml --load _model.pt &> ${RUN_NAME}.log
#python -m torch.distributed.launch --nproc_per_node=3 --use_env train_mp_enet.py --train_def train_enet.yml --model_def model_enet.yml --load _model.pt &> ${RUN_NAME}.log

#python -m torch.distributed.launch --nproc_per_node=8 --use_env train_mp.py --train_def train.yml --model_def model.yml --save ${RUN_NAME}_qenet.pt &> ${RUN_NAME}_qenet.log
#python pt_split.py model.yml ${RUN_NAME}_qenet.pt ${RUN_NAME}_qnet.pt ${RUN_NAME}_qnet.jpt ${RUN_NAME}_enet.pt ${RUN_NAME}_enet.jpt
#export QNET_JPT=${RUN_NAME}_qnet.jpt
#python -m torch.distributed.launch --nproc_per_node=8 --use_env train_mp_enet.py --train_def train_enet.yml --model_def model_enet.yml --load ${RUN_NAME}_enet.pt --save ${RUN_NAME}_enet_st.pt &> ${RUN_NAME}_enet.log

