#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
# run this file in the ReVL directory
DIR=`pwd`
export TRANSFORMERS_CACHE=/ocean/projects/cis240092p/amartin1/.cache/huggingface/hub

# GPUS_PER_NODE=1
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

K=3
EPOCHS=1
# TRAINING_EPOCHS=$((EPOCHS * (K + 1)))
TRAINING_EPOCHS=5

# MODEL="Qwen/Qwen-VL-Chat" #"Qwen/Qwen-VL-Chat"/"Qwen/Qwen-VL"  Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
# DATA="data/json_data/tmp_qwen2_test_data.json"
DATA="data/json_data/ReVL_text_to_point_10000_with_augmented_images.json"
OUT="finetune/qwen2testout"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

module load anaconda3
module load cuda
module load nvhpc
module load gcc

source .qwen-venv/bin/activate

wandb login f9ac323d3bf6a5c10af396c0d1b76cf08161d409

nvidia-smi

python finetune/finetune_qwenvl2.py \
    --train_dataset $DATA \
    --output_dir $OUT\
    --epochs $TRAINING_EPOCHS