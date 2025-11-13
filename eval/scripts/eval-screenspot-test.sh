#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

export CUDA_VISIBLE_DEVICES=0

# Defaults (can be overridden via CLI args)
MODEL="/ocean/projects/cis240092p/amartin1/ReVL/finetune/model_output/Qwen/Qwen2.5-VL-3B-Instruct_all_tasks_revl_1000%100_K1_20251024_032530"
K="1"
DATA="screenspot_web"
CONTEXT="True"

# Usage helper
usage() {
    echo "Usage: $0 [--model NAME|-m NAME] [--k K|-k K] [--data NAME|-d NAME] [--context True|False|-c True|False]"
    exit 1
}

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL="$2"; shift 2 ;;
        -k|--k)
            K="$2"; shift 2 ;;
        -d|--data)
            DATA="$2"; shift 2 ;;
        -c|--context)
            CONTEXT="$2"; shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "Unknown option: $1"; usage ;;
    esac
done

module load anaconda3
module load cuda
module load nvhpc
module load gcc
module load ffmpeg

conda activate python312

source /ocean/projects/cis240092p/amartin1/ReVL/.qwen_py312/bin/activate

python /ocean/projects/cis240092p/amartin1/ReVL/eval/python_scripts/eval.py --data "$DATA" --model "$MODEL" --k "$K" --context "$CONTEXT"