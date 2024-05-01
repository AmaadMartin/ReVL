#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`


MODEL="Qwen/Qwen-VL-Chat" #"Qwen/Qwen-VL-Chat"/"Qwen/Qwen-VL" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="updated_small_train.json"

export CUDA_VISIBLE_DEVICES=0

module load anaconda3
module load cuda
module load nvhpc

conda activate py310

source env/bin/activate

export WANDB_PROJECT=revl_baseline

wandb login d73de72f4a6e9d226499f9c6da0c361a04336fde


python finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 False \
    --fp16 False \
    --fix_vit True \
    --output_dir output_qwen \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 700 \
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
    --gradient_checkpointing \
    --use_lora 