#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

export CUDA_VISIBLE_DEVICES=0

module load anaconda3
module load cuda
module load nvhpc

conda activate py310

source env/bin/activate

wandb login d73de72f4a6e9d226499f9c6da0c361a04336fde

python eval/eval.py --data screenspot_bbox_test_text --model checkpoint-2750