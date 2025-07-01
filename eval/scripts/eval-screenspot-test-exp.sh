#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

export CUDA_VISIBLE_DEVICES=0

module load anaconda3
module load cuda
module load nvhpc

conda activate py310

source env/bin/activate

wandb login f9ac323d3bf6a5c10af396c0d1b76cf08161d409

python eval/experiment_eval.py --data_name screenspot_bbox_test --model line_experiment_out --k 3 --exp_mode 2  