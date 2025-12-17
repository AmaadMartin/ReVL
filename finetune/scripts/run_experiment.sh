#!/bin/bash
#
# ReVL Unified Experiment Runner Shell Script
#
# This script provides a convenient interface to run experiments that alternate
# between training and evaluation, logging everything to a single wandb run.
#
# Usage:
#   ./run_experiment.sh --eval_steps 1000 --eval_data screenspot_web [other args]
#
# Required arguments:
#   --eval_steps N         Run evaluation every N training steps
#   --eval_data DATASET    Dataset to use for evaluation
#
# Training arguments (from sft.sh):
#   --gpus_per_node N      Number of GPUs per node (default: 1)
#   --nnodes N             Number of nodes (default: 1)
#   --node_rank R          Node rank (default: 0)
#   --master_addr HOST     Master address (default: localhost)
#   --master_port PORT     Master port (default: 6001)
#   --deepspeed PATH       DeepSpeed config path
#   --llm HF_ID            HuggingFace model ID (default: Qwen/Qwen2.5-VL-7B-Instruct)
#   --lr LR                Learning rate (default: 2e-7)
#   --batch_size BS        Per-device batch size (default: 4)
#   --grad_accum_steps N   Gradient accumulation steps (default: 4)
#   --datasets PATHS       Training dataset(s)
#   --total_steps N        Total training steps (-1 for full epoch)
#
# ReVL parameters:
#   --K N                  Number of ReVL partition steps (default: 1)
#   --context BOOL         Keep context across turns (default: True)
#   --resolution MODE      Resolution: dynamic or static (default: dynamic)
#   --demarcation MODE     Demarcation: none, lines, or quadrants (default: none)
#
# WandB:
#   --wandb_project NAME   WandB project name (default: ReVL_experiment)
#   --wandb_run_id ID      Resume a specific WandB run ID
#
# Extra arguments are passed through to the training script.

export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# Default values
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

# DeepSpeed configuration
DEEPSPEED=/ocean/projects/cis240092p/amartin1/ReVL/finetune/python_scripts/Qwen3-VL_10_23_25/qwen-vl-finetune/scripts/zero3.json

# Model configuration
LLM="Qwen/Qwen2.5-VL-7B-Instruct"

# Training hyperparameters
LR=2e-7
BATCH_SIZE=4
GRAD_ACCUM_STEPS=4
DATASETS="all_tasks_revl_10000%100"
TOTAL_STEPS=-1

# ReVL parameters
K=1
CONTEXT="True"
RESOLUTION="dynamic"
DEMARCATION="none"

# Required experiment parameters (no defaults)
EVAL_STEPS=""
EVAL_DATA=""

# WandB
WANDB_PROJECT="ReVL_experiment"
WANDB_RUN_ID=""

# LoRA
LORA_ENABLE="True"

# Extra args for passthrough
EXTRA_ARGS=()

# Usage function
usage() {
    echo "Usage: $0 --eval_steps N --eval_data DATASET [options]"
    echo ""
    echo "Required arguments:"
    echo "  --eval_steps N         Run evaluation every N training steps"
    echo "  --eval_data DATASET    Dataset to use for evaluation (e.g., screenspot_web)"
    echo ""
    echo "Training arguments:"
    echo "  --gpus_per_node N      Number of GPUs per node (default: $GPUS_PER_NODE)"
    echo "  --nnodes N             Number of nodes (default: $NNODES)"
    echo "  --llm HF_ID            HuggingFace model ID (default: $LLM)"
    echo "  --lr LR                Learning rate (default: $LR)"
    echo "  --batch_size BS        Batch size (default: $BATCH_SIZE)"
    echo "  --grad_accum_steps N   Gradient accumulation steps (default: $GRAD_ACCUM_STEPS)"
    echo "  --datasets PATHS       Training dataset(s) (default: $DATASETS)"
    echo "  --total_steps N        Total training steps, -1 for full epoch (default: $TOTAL_STEPS)"
    echo ""
    echo "ReVL parameters:"
    echo "  --K N                  ReVL partition steps (default: $K)"
    echo "  --context BOOL         Keep context (default: $CONTEXT)"
    echo "  --resolution MODE      dynamic or static (default: $RESOLUTION)"
    echo "  --demarcation MODE     none, lines, or quadrants (default: $DEMARCATION)"
    echo ""
    echo "WandB:"
    echo "  --wandb_project NAME   Project name (default: $WANDB_PROJECT)"
    echo "  --wandb_run_id ID      Resume specific run ID"
    echo ""
    echo "Extra arguments are passed to the training script."
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        # Required experiment parameters
        --eval_steps)
            EVAL_STEPS="$2"; shift 2 ;;
        --eval_data)
            EVAL_DATA="$2"; shift 2 ;;
        
        # Distributed training
        --gpus_per_node)
            GPUS_PER_NODE="$2"; shift 2 ;;
        --nnodes)
            NNODES="$2"; shift 2 ;;
        --node_rank)
            NODE_RANK="$2"; shift 2 ;;
        --master_addr)
            MASTER_ADDR="$2"; shift 2 ;;
        --master_port)
            MASTER_PORT="$2"; shift 2 ;;
        --deepspeed)
            DEEPSPEED="$2"; shift 2 ;;
        
        # Model configuration
        --llm)
            LLM="$2"; shift 2 ;;
        
        # Training hyperparameters
        --lr)
            LR="$2"; shift 2 ;;
        --batch_size)
            BATCH_SIZE="$2"; shift 2 ;;
        --grad_accum_steps)
            GRAD_ACCUM_STEPS="$2"; shift 2 ;;
        --datasets)
            DATASETS="$2"; shift 2 ;;
        --total_steps)
            TOTAL_STEPS="$2"; shift 2 ;;
        --lora_enable)
            LORA_ENABLE="$2"; shift 2 ;;
        
        # ReVL parameters
        --K)
            K="$2"; shift 2 ;;
        --context)
            CONTEXT="$2"; shift 2 ;;
        --resolution)
            RESOLUTION="$2"; shift 2 ;;
        --demarcation)
            DEMARCATION="$2"; shift 2 ;;
        
        # WandB
        --wandb_project)
            WANDB_PROJECT="$2"; shift 2 ;;
        --wandb_run_id)
            WANDB_RUN_ID="$2"; shift 2 ;;
        
        # Help
        -h|--help)
            usage ;;
        
        # Unknown args - pass through to Python script
        *)
            EXTRA_ARGS+=("$1")
            # If next token is a value (doesn't start with --), include it too
            if [[ -n "$2" && "$2" != --* ]]; then
                EXTRA_ARGS+=("$2")
                shift 2
            else
                shift 1
            fi
            ;;
    esac
done

# Validate required arguments
if [[ -z "$EVAL_STEPS" ]]; then
    echo "Error: --eval_steps is required"
    usage
fi

if [[ -z "$EVAL_DATA" ]]; then
    echo "Error: --eval_data is required"
    usage
fi

# Validate resolution
case "${RESOLUTION}" in
    dynamic|Dynamic|static|Static)
        ;;
    *)
        echo "Invalid --resolution '${RESOLUTION}'. Use 'dynamic' or 'static'." >&2
        exit 1
        ;;
esac

# Validate demarcation
case "${DEMARCATION}" in
    none|lines|quadrants)
        ;;
    *)
        echo "Invalid --demarcation '${DEMARCATION}'. Use 'none', 'lines', or 'quadrants'." >&2
        exit 1
        ;;
esac

# Print configuration
echo "============================================================"
echo "ReVL UNIFIED EXPERIMENT RUNNER"
echo "============================================================"
echo "Eval Steps: $EVAL_STEPS"
echo "Eval Data: $EVAL_DATA"
echo "LLM: $LLM"
echo "Datasets: $DATASETS"
echo "K: $K"
echo "Resolution: $RESOLUTION"
echo "Context: $CONTEXT"
echo "Demarcation: $DEMARCATION"
echo "Learning Rate: $LR"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation: $GRAD_ACCUM_STEPS"
echo "Total Steps: $TOTAL_STEPS"
echo "GPUs per Node: $GPUS_PER_NODE"
echo "Nodes: $NNODES"
echo "WandB Project: $WANDB_PROJECT"
if [[ -n "$WANDB_RUN_ID" ]]; then
    echo "WandB Run ID: $WANDB_RUN_ID"
fi
echo "============================================================"

# Load modules (for HPC environment)
module load anaconda3
module load cuda
module load nvhpc
module load gcc
module load ffmpeg

conda activate python312

source /ocean/projects/cis240092p/amartin1/ReVL/.qwen_py312/bin/activate

# Build Python command
PYTHON_SCRIPT=/ocean/projects/cis240092p/amartin1/ReVL/finetune/python_scripts/run_experiment.py

PYTHON_ARGS=(
    "--eval_steps" "$EVAL_STEPS"
    "--eval_data" "$EVAL_DATA"
    "--llm" "$LLM"
    "--datasets" "$DATASETS"
    "--K" "$K"
    "--resolution" "$RESOLUTION"
    "--context" "$CONTEXT"
    "--demarcation" "$DEMARCATION"
    "--lr" "$LR"
    "--batch_size" "$BATCH_SIZE"
    "--grad_accum_steps" "$GRAD_ACCUM_STEPS"
    "--total_steps" "$TOTAL_STEPS"
    "--gpus_per_node" "$GPUS_PER_NODE"
    "--nnodes" "$NNODES"
    "--node_rank" "$NODE_RANK"
    "--master_addr" "$MASTER_ADDR"
    "--master_port" "$MASTER_PORT"
    "--deepspeed" "$DEEPSPEED"
    "--wandb_project" "$WANDB_PROJECT"
    "--lora_enable" "$LORA_ENABLE"
)

# Add optional wandb_run_id if provided
if [[ -n "$WANDB_RUN_ID" ]]; then
    PYTHON_ARGS+=("--wandb_run_id" "$WANDB_RUN_ID")
fi

# Add extra args
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    PYTHON_ARGS+=("${EXTRA_ARGS[@]}")
fi

# Run the experiment
echo "Running: python $PYTHON_SCRIPT ${PYTHON_ARGS[*]}"
python "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}"

