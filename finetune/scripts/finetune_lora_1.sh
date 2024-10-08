#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="Qwen/Qwen-VL-Chat" #"Qwen/Qwen-VL-Chat"/"Qwen/Qwen-VL"  Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="json_data/small_seeclick_train_data_unaugmented.json"
OUT="finetune/output"

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

conda activate py310

source env/bin/activate

wandb login d73de72f4a6e9d226499f9c6da0c361a04336fde

torchrun $DISTRIBUTED_ARGS finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 False \
    --fix_vit True \
    --output_dir $OUT \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 60 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "wandb" \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --use_lora \
    --gradient_checkpointing \
    --deepspeed Qwen-VL/finetune/ds_config_zero2.json