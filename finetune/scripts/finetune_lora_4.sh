#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`
export TRANSFORMERS_CACHE=/ocean/projects/cis240092p/amartin1/.cache/huggingface/hub

GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

K=3
EPOCHS=1
# TRAINING_EPOCHS=$((EPOCHS * (K + 1)))
TRAINING_EPOCHS=1

MODEL="Qwen/Qwen-VL-Chat" #"Qwen/Qwen-VL-Chat"/"Qwen/Qwen-VL"  Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="data/json_data/SeeClick_model_train_data.json"
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

torchrun $DISTRIBUTED_ARGS finetune/finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 False \
    --fix_vit True \
    --output_dir $OUT \
    --num_train_epochs $TRAINING_EPOCHS \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 1200 \
    --save_strategy "steps" \
    --save_steps 250 \
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
    --deepspeed finetune/config/ds_config_zero2.json