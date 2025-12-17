#!/usr/bin/env python3
"""
Unified Experiment Runner for ReVL

This script orchestrates training and evaluation cycles, managing:
- Alternating between finetuning and evaluation
- Automatic checkpoint discovery and resume
- Unified wandb logging across train and eval
- Easy experiment configuration via command line

Usage:
    python run_experiment.py --experiment_id <id> --eval_steps <steps> --eval_data <dataset> [training args] [eval args]
"""

import os
import sys
import re
import glob
import json
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import wandb


# Base paths
BASE_DIR = Path("/ocean/projects/cis240092p/amartin1/ReVL")
FINETUNE_DIR = BASE_DIR / "finetune"
MODEL_OUTPUT_DIR = FINETUNE_DIR / "model_output"
TRAIN_SCRIPT = FINETUNE_DIR / "python_scripts/Qwen3-VL_10_23_25/qwen-vl-finetune/qwenvl/train/train_qwen.py"
EVAL_SCRIPT = BASE_DIR / "eval/python_scripts/eval.py"
DEEPSPEED_CONFIG = FINETUNE_DIR / "python_scripts/Qwen3-VL_10_23_25/qwen-vl-finetune/scripts/zero3.json"


def get_experiment_pattern(
    llm: str,
    datasets: str,
    K: int,
    resolution: str,
    context: bool,
    demarcation: str
) -> str:
    """
    Generate a pattern to match experiment directories based on settings.
    The pattern excludes timestamp to find all runs with matching settings.
    """
    # Normalize the LLM name (e.g., "Qwen/Qwen2.5-VL-7B-Instruct" -> "Qwen2.5-VL-7B-Instruct")
    llm_name = llm.split("/")[-1]
    dataset_name = os.path.basename(datasets)
    ctx_str = str(context)
    
    # Pattern: {llm}_{dataset}_K{K}_res-{resolution}_ctx-{context}_demarc-{demarcation}_*
    pattern = f"{llm_name}_{dataset_name}_K{K}_res-{resolution}_ctx-{ctx_str}_demarc-{demarcation}_*"
    return pattern


def get_experiment_name(
    llm: str,
    datasets: str,
    K: int,
    resolution: str,
    context: bool,
    demarcation: str,
    timestamp: Optional[str] = None
) -> str:
    """
    Generate a unique experiment name based on settings.
    """
    llm_name = llm.split("/")[-1]
    dataset_name = os.path.basename(datasets)
    ctx_str = str(context)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{llm_name}_{dataset_name}_K{K}_res-{resolution}_ctx-{ctx_str}_demarc-{demarcation}_{timestamp}"


def find_latest_matching_directory(
    llm: str,
    datasets: str,
    K: int,
    resolution: str,
    context: bool,
    demarcation: str
) -> Optional[Path]:
    """
    Find the most recent model directory matching the experiment settings.
    Returns None if no matching directory exists.
    """
    pattern = get_experiment_pattern(llm, datasets, K, resolution, context, demarcation)
    
    # Look for directories matching the pattern
    # The directories are organized as MODEL_OUTPUT_DIR/Qwen/{experiment_name}
    llm_name = llm.split("/")[-1]
    llm_base = llm.split("/")[0] if "/" in llm else "models"
    
    search_path = MODEL_OUTPUT_DIR / llm_base / pattern
    matching_dirs = glob.glob(str(search_path))
    
    if not matching_dirs:
        print(f"No existing directories found matching pattern: {search_path}")
        return None
    
    # Sort by timestamp (last part of directory name)
    def extract_timestamp(path: str) -> str:
        name = os.path.basename(path)
        # Timestamp is the last part after the last underscore pair (YYYYMMDD_HHMMSS)
        parts = name.split("_")
        if len(parts) >= 2:
            return "_".join(parts[-2:])
        return "00000000_000000"
    
    matching_dirs.sort(key=extract_timestamp, reverse=True)
    latest = Path(matching_dirs[0])
    print(f"Found latest matching directory: {latest}")
    return latest


def find_latest_checkpoint(model_dir: Path) -> Optional[Path]:
    """
    Find the latest checkpoint in a model directory.
    Checkpoints are named checkpoint-{step}.
    """
    checkpoint_dirs = glob.glob(str(model_dir / "checkpoint-*"))
    
    if not checkpoint_dirs:
        # Check if this is a completed training (final model saved at root)
        if (model_dir / "adapter_model.safetensors").exists() or \
           (model_dir / "model.safetensors").exists() or \
           (model_dir / "pytorch_model.bin").exists():
            print(f"Found completed model at: {model_dir}")
            return model_dir
        return None
    
    # Sort by step number
    def extract_step(path: str) -> int:
        match = re.search(r'checkpoint-(\d+)', path)
        return int(match.group(1)) if match else 0
    
    checkpoint_dirs.sort(key=extract_step, reverse=True)
    latest = Path(checkpoint_dirs[0])
    print(f"Found latest checkpoint: {latest}")
    return latest


def get_current_step(model_dir: Path) -> int:
    """
    Get the current training step from trainer_state.json or checkpoint name.
    """
    # Try to read from trainer_state.json
    trainer_state_path = model_dir / "trainer_state.json"
    if trainer_state_path.exists():
        try:
            with open(trainer_state_path, 'r') as f:
                state = json.load(f)
                return state.get('global_step', 0)
        except Exception as e:
            print(f"Warning: Could not read trainer_state.json: {e}")
    
    # Fall back to extracting from checkpoint directory name
    latest_ckpt = find_latest_checkpoint(model_dir)
    if latest_ckpt:
        match = re.search(r'checkpoint-(\d+)', str(latest_ckpt))
        if match:
            return int(match.group(1))
    
    return 0


def get_max_steps(model_dir: Path) -> int:
    """
    Get the total number of training steps from training_args.
    """
    trainer_state_path = model_dir / "trainer_state.json"
    if trainer_state_path.exists():
        try:
            with open(trainer_state_path, 'r') as f:
                state = json.load(f)
                return state.get('max_steps', -1)
        except Exception:
            pass
    return -1


def run_training(
    output_dir: Path,
    llm: str,
    datasets: str,
    K: int,
    resolution: str,
    context: bool,
    demarcation: str,
    lr: float,
    batch_size: int,
    grad_accum_steps: int,
    max_steps: int,
    save_steps: int,
    run_name: str,
    wandb_run_id: str,
    extra_args: List[str],
    gpus_per_node: int = 1,
    nnodes: int = 1,
    node_rank: int = 0,
    master_addr: str = "localhost",
    master_port: int = 6001,
    deepspeed_config: Optional[str] = None,
    lora_enable: bool = True,
) -> int:
    """
    Run the training script with the given configuration.
    Returns the exit code of the training process.
    """
    if deepspeed_config is None:
        deepspeed_config = str(DEEPSPEED_CONFIG)
    
    # Resolution settings
    if resolution.lower() == "dynamic":
        max_pixels = 50176
        min_pixels = 784
    elif resolution.lower() == "static":
        px = 448 * 448
        max_pixels = px
        min_pixels = px
    else:
        raise ValueError(f"Invalid resolution: {resolution}")
    
    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={gpus_per_node}",
        f"--nnodes={nnodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        str(TRAIN_SCRIPT),
        "--deepspeed", deepspeed_config,
        "--model_name_or_path", llm,
        "--dataset_use", datasets,
        "--data_flatten", "True",
        "--tune_mm_vision", "False",
        "--tune_mm_mlp", "True",
        "--tune_mm_llm", "True",
        "--bf16",
        "--output_dir", str(output_dir),
        "--max_steps", str(max_steps),
        "--per_device_train_batch_size", str(batch_size),
        "--per_device_eval_batch_size", str(batch_size * 2),
        "--gradient_accumulation_steps", str(grad_accum_steps),
        "--max_pixels", str(max_pixels),
        "--min_pixels", str(min_pixels),
        "--eval_strategy", "no",
        "--save_strategy", "steps",
        "--save_steps", str(save_steps),
        "--save_total_limit", "2",
        "--learning_rate", str(lr),
        "--weight_decay", "0",
        "--warmup_ratio", "0.03",
        "--max_grad_norm", "1",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--model_max_length", "8192",
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "4",
        "--run_name", run_name,
        "--report_to", "wandb",
        "--data_packing", "False",
        "--K", str(K),
        "--demarcation", demarcation,
        "--context", str(context),
        "--lora_enable", str(lora_enable),
    ]
    
    # Add extra args
    cmd.extend(extra_args)
    
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Max steps: {max_steps}")
    print(f"Save steps: {save_steps}")
    print(f"WandB run ID: {wandb_run_id}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    # Set wandb environment variables for resume
    env = os.environ.copy()
    env["WANDB_RUN_ID"] = wandb_run_id
    env["WANDB_RESUME"] = "allow"
    
    # Run training
    result = subprocess.run(cmd, env=env, shell=True) # replace with run_in_venv function (Takes command and constructs bash script to run)
    return result.returncode


def run_evaluation(
    model_path: Path,
    eval_data: str,
    K: int,
    context: bool,
    resolution: str,
    demarcation: str,
    run_name: str,
    wandb_run_id: str,
    step: int,
) -> Tuple[int, Optional[float]]:
    """
    Run the evaluation script with the given configuration.
    Returns (exit_code, accuracy).
    """
    print(f"\n{'='*60}")
    print("STARTING EVALUATION")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    print(f"Eval data: {eval_data}")
    print(f"Step: {step}")
    print(f"{'='*60}\n")
    
    # Set wandb environment to continue the same run
    env = os.environ.copy()
    env["WANDB_RUN_ID"] = wandb_run_id
    env["WANDB_RESUME"] = "must"
    
    # We need to modify eval.py to accept wandb_run_id or we run eval inline
    # For now, we'll run eval inline using the eval module
    
    # Add the eval directory to path
    eval_dir = BASE_DIR / "eval/python_scripts"
    sys.path.insert(0, str(eval_dir))
    
    # Import evaluation components
    try:
        from qwen2_vl.model import Qwen2VLChat
        from eval import eval as eval_fn, TEST_DATA_PATH
        import json as json_module
        
        # Resume wandb run
        wandb.init(
            project="ReVL_experiment",
            id=wandb_run_id,
            resume="must",
        )
        
        # Resolution settings
        if resolution.lower() == "dynamic":
            max_pixels = 50176
            min_pixels = 784
        else:
            px = 448 * 448
            max_pixels = px
            min_pixels = px
        
        print(f"Loading model from {model_path}")
        model = Qwen2VLChat(
            model_path=str(model_path),
            temperature=0.01,
            top_p=0.001,
            top_k=1,
            use_custom_prompt=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        
        # Load test data
        test_data_path = TEST_DATA_PATH.format(data_name=eval_data)
        with open(test_data_path, 'r') as f:
            test_data = json_module.load(f)
        
        # Run evaluation
        accuracy = eval_fn(
            model, 
            None, 
            test_data, 
            visualize=True, 
            k=K, 
            keep_context=context, 
            demarcation=demarcation
        )
        
        # Log evaluation results with step prefix
        wandb.log({
            f"eval/{eval_data}/accuracy": accuracy,
            f"eval/{eval_data}/step": step,
            "eval_step": step,
        })
        
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE - Accuracy: {accuracy:.4f}")
        print(f"{'='*60}\n")
        
        # Clean up model to free GPU memory
        del model
        import torch
        torch.cuda.empty_cache()
        
        return 0, accuracy
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1, None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Experiment Runner for ReVL - alternates training and evaluation"
    )
    
    # Experiment control
    parser.add_argument(
        "--eval_steps", type=int, required=True,
        help="Run evaluation every N training steps"
    )
    parser.add_argument(
        "--eval_data", type=str, required=True,
        help="Dataset to use for evaluation (e.g., screenspot_web)"
    )
    parser.add_argument(
        "--total_steps", type=int, default=-1,
        help="Total training steps (-1 for full epoch training)"
    )
    
    # Model configuration
    parser.add_argument("--llm", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="HuggingFace model ID")
    parser.add_argument("--datasets", type=str, default="all_tasks_revl_10000%100",
                       help="Training dataset")
    
    # ReVL parameters
    parser.add_argument("--K", type=int, default=1,
                       help="Number of ReVL partition steps")
    parser.add_argument("--resolution", type=str, default="dynamic",
                       choices=["dynamic", "static"],
                       help="Image resolution strategy")
    parser.add_argument("--context", type=str, default="True",
                       help="Whether to keep context across turns")
    parser.add_argument("--demarcation", type=str, default="none",
                       choices=["none", "lines", "quadrants"],
                       help="Demarcation strategy")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=2e-7,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Per-device batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--lora_enable", type=str, default="True",
                       help="Enable LoRA training")
    
    # Distributed training
    parser.add_argument("--gpus_per_node", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=int, default=6001)
    parser.add_argument("--deepspeed", type=str, default=None,
                       help="Path to DeepSpeed config")
    
    # WandB configuration
    parser.add_argument("--wandb_project", type=str, default="ReVL_experiment",
                       help="WandB project name")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                       help="Resume a specific WandB run ID")
    
    # Parse known args and collect extra args for trainer
    args, extra_args = parser.parse_known_args()
    args.extra_args = extra_args
    
    # Convert string bools
    args.context = args.context.lower() in ('true', '1', 'yes')
    args.lora_enable = args.lora_enable.lower() in ('true', '1', 'yes')
    
    return args


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("ReVL UNIFIED EXPERIMENT RUNNER")
    print(f"{'='*60}")
    print(f"LLM: {args.llm}")
    print(f"Dataset: {args.datasets}")
    print(f"K: {args.K}")
    print(f"Resolution: {args.resolution}")
    print(f"Context: {args.context}")
    print(f"Demarcation: {args.demarcation}")
    print(f"Eval Data: {args.eval_data}")
    print(f"Eval Steps: {args.eval_steps}")
    print(f"{'='*60}\n")
    
    # Find or create experiment directory
    existing_dir = find_latest_matching_directory(
        args.llm, args.datasets, args.K, 
        args.resolution, args.context, args.demarcation
    )
    
    if existing_dir:
        output_dir = existing_dir
        current_step = get_current_step(output_dir)
        # Extract timestamp from existing directory name
        dir_name = output_dir.name
        parts = dir_name.split("_")
        timestamp = "_".join(parts[-2:])
        print(f"Resuming from existing experiment at step {current_step}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = get_experiment_name(
            args.llm, args.datasets, args.K,
            args.resolution, args.context, args.demarcation,
            timestamp
        )
        llm_base = args.llm.split("/")[0] if "/" in args.llm else "models"
        output_dir = MODEL_OUTPUT_DIR / llm_base / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        current_step = 0
        print(f"Created new experiment directory: {output_dir}")
    
    # Generate run name and wandb run ID
    run_name = get_experiment_name(
        args.llm, args.datasets, args.K,
        args.resolution, args.context, args.demarcation,
        timestamp
    )
    
    # Use provided wandb_run_id or generate from experiment name
    if args.wandb_run_id:
        wandb_run_id = args.wandb_run_id
    else:
        # Create a stable ID based on experiment settings (without timestamp for continuity)
        llm_name = args.llm.split("/")[-1]
        dataset_name = os.path.basename(args.datasets)
        wandb_run_id = f"{llm_name}_{dataset_name}_K{args.K}_{args.resolution}_{args.context}_{args.demarcation}"
        # Sanitize for wandb
        wandb_run_id = re.sub(r'[^a-zA-Z0-9_-]', '_', wandb_run_id)
    
    print(f"WandB Run ID: {wandb_run_id}")
    
    # Initialize wandb for the main process
    wandb.init(
        project=args.wandb_project,
        id=wandb_run_id,
        resume="allow",
        name=run_name,
        config={
            "llm": args.llm,
            "datasets": args.datasets,
            "K": args.K,
            "resolution": args.resolution,
            "context": args.context,
            "demarcation": args.demarcation,
            "eval_data": args.eval_data,
            "eval_steps": args.eval_steps,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "lora_enable": args.lora_enable,
        }
    )
    
    # Calculate total steps if not specified
    total_steps = args.total_steps if args.total_steps > 0 else 100000  # Large default
    
    # Main training/eval loop
    next_eval_step = ((current_step // args.eval_steps) + 1) * args.eval_steps
    
    while current_step < total_steps:
        # Calculate how many steps to train until next eval
        steps_until_eval = min(next_eval_step - current_step, total_steps - current_step)
        target_step = current_step + steps_until_eval
        
        if steps_until_eval > 0:
            print(f"\nTraining from step {current_step} to {target_step}...")
            
            # Run training
            exit_code = run_training(
                output_dir=output_dir,
                llm=args.llm,
                datasets=args.datasets,
                K=args.K,
                resolution=args.resolution,
                context=args.context,
                demarcation=args.demarcation,
                lr=args.lr,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps,
                max_steps=target_step,
                save_steps=args.eval_steps,  # Save at eval points
                run_name=run_name,
                wandb_run_id=wandb_run_id,
                extra_args=args.extra_args,
                gpus_per_node=args.gpus_per_node,
                nnodes=args.nnodes,
                node_rank=args.node_rank,
                master_addr=args.master_addr,
                master_port=int(args.master_port),
                deepspeed_config=args.deepspeed,
                lora_enable=args.lora_enable,
            )
            
            if exit_code != 0:
                print(f"Training failed with exit code {exit_code}")
                wandb.finish(exit_code=1)
                return exit_code
            
            current_step = target_step
        
        # Run evaluation
        checkpoint_path = find_latest_checkpoint(output_dir)
        if checkpoint_path:
            eval_exit_code, accuracy = run_evaluation(
                model_path=checkpoint_path,
                eval_data=args.eval_data,
                K=args.K,
                context=args.context,
                resolution=args.resolution,
                demarcation=args.demarcation,
                run_name=run_name,
                wandb_run_id=wandb_run_id,
                step=current_step,
            )
            
            if eval_exit_code != 0:
                print(f"Warning: Evaluation failed with exit code {eval_exit_code}")
        else:
            print(f"Warning: No checkpoint found for evaluation at step {current_step}")
        
        next_eval_step += args.eval_steps
        
        # Check if training is complete
        actual_step = get_current_step(output_dir)
        max_steps = get_max_steps(output_dir)
        if max_steps > 0 and actual_step >= max_steps:
            print(f"\nTraining complete at step {actual_step}")
            break
    
    # Final evaluation
    print("\nRunning final evaluation...")
    checkpoint_path = find_latest_checkpoint(output_dir)
    if checkpoint_path:
        eval_exit_code, accuracy = run_evaluation(
            model_path=checkpoint_path,
            eval_data=args.eval_data,
            K=args.K,
            context=args.context,
            resolution=args.resolution,
            demarcation=args.demarcation,
            run_name=run_name,
            wandb_run_id=wandb_run_id,
            step=current_step,
        )
        
        if accuracy is not None:
            wandb.log({"final_accuracy": accuracy})
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Final step: {current_step}")
    print(f"WandB Run ID: {wandb_run_id}")
    print(f"{'='*60}\n")
    
    wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())

