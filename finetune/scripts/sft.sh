#!/bin/bash

# Distributed training configuration
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
# NNODES=${WORLD_SIZE:-1}
# NPROC_PER_NODE=1

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# DeepSpeed configuration
deepspeed=finetune/Qwen2.5-VL/qwen-vl-finetune/scripts/zero3.json

# Model configuration
llm=Qwen/Qwen2.5-VL-3B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=2e-7
batch_size=4
grad_accum_steps=4
K=3

# Training entry point
entry_file=finetune/Qwen2.5-VL/qwen-vl-finetune/qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=/ocean/projects/cis240092p/amartin1/ReVL/data/json_data/lazy_ReVL_text_to_point_100000.json

# Output configuration
run_name="qwen2vl-baseline"
output_dir=finetune/Qwen2.5-VL/qwen-vl-finetune/output

# Parse command line overrides and passthrough args for the trainer
USER_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus_per_node) GPUS_PER_NODE="$2"; shift 2 ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --node_rank) NODE_RANK="$2"; shift 2 ;;
        --master_addr) MASTER_ADDR="$2"; shift 2 ;;
        --master_port) MASTER_PORT="$2"; shift 2 ;;
        --deepspeed) deepspeed="$2"; shift 2 ;;
        --llm) llm="$2"; shift 2 ;;
        --lr) lr="$2"; shift 2 ;;
        --batch_size) batch_size="$2"; shift 2 ;;
        --grad_accum_steps) grad_accum_steps="$2"; shift 2 ;;
        --entry_file) entry_file="$2"; shift 2 ;;
        --datasets) datasets="$2"; shift 2 ;;
        --run_name) run_name="$2"; shift 2 ;;
        --output_dir) output_dir="$2"; shift 2 ;;
        --K) K="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--gpus_per_node N] [--nnodes N] [--node_rank R] [--master_addr HOST] [--master_port PORT] \
    [--deepspeed PATH] [--llm HF_ID] [--lr LR] [--batch_size BS] [--grad_accum_steps STEPS] \
    [--entry_file PATH] [--datasets PATHS] [--run_name NAME] [--output_dir DIR] [--K N] [trainer args ...]"
            echo "Any unrecognized flags are forwarded to the Python trainer and override defaults."
            exit 0 ;;
        *)
            USER_ARGS+=("$1")
            # If next token is a value (doesn't start with --), include it too
            if [[ -n "$2" && "$2" != --* ]]; then
                USER_ARGS+=("$2")
                shift 2
            else
                shift 1
            fi
            ;;
    esac
done

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb \
    --K ${K}"

# Append any user-provided extra trainer args at the end so they take precedence
if [[ ${#USER_ARGS[@]} -gt 0 ]]; then
    args+=" \
    ${USER_ARGS[*]}"
fi

module load anaconda3
module load cuda
module load nvhpc
module load gcc
module load ffmpeg

conda activate python312

source .qwen_py312/bin/activate

# Launch training
torchrun $DISTRIBUTED_ARGS \
         ${entry_file} ${args}