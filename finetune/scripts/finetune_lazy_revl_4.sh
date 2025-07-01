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

EVAL_DATA_SIZE=1000
TRAIN_DATA_SIZE=10000


MODEL="Qwen/Qwen2-VL-7B-Instruct"
DATA="data/json_data/lazy_ReVL_all_tasks_1000000.json"
OUT="finetune/output/qwen2-7b-instruct-trl-sft-lazy_ReVL_all_tasks_10000"

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

torchrun $DISTRIBUTED_ARGS finetune/ReVL_lazy_finetune.py \
    --model_name_or_path $MODEL \
    --train_data_path $DATA \
    --output_dir $OUT \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --max_grad_norm 0.3 \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --eval_steps 10 \
    --save_steps 20 \
    --K $K \
    --train_data_size $TRAIN_DATA_SIZE \
    --eval_data_size $EVAL_DATA_SIZE