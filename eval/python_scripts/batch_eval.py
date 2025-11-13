#!/usr/bin/env python3
"""
Script to batch evaluate models that haven't been evaluated yet.
Checks wandb for existing runs and only evaluates models not present.
Submits evaluation jobs to a GPU cluster using sbatch.

Usage:
    # Basic usage with auto-detection (extracts K and context from model names):
    python batch_eval.py --auto_detect_k --auto_detect_context
    
    # Dry run to see what would be evaluated:
    python batch_eval.py --auto_detect_k --auto_detect_context --dry_run
    
    # Manual mode - evaluate specific k values on all models:
    python batch_eval.py --k_values "0,1,2"
    
    # Manual mode - evaluate only with context enabled:
    python batch_eval.py --context_values "true"
    
    # Use a different dataset:
    python batch_eval.py --dataset my_dataset_name --auto_detect_k --auto_detect_context
    
    # Specify wandb entity (team/username):
    python batch_eval.py --entity athena-innovations --auto_detect_k --auto_detect_context
    
    # Custom sbatch settings:
    python batch_eval.py --sbatch_partition GPU-shared --sbatch_time "24:00:00" --sbatch_gpus "h100-80:1"

The script will:
1. Query wandb for all existing run names in the specified project
2. Find all model directories in the model output folder (excludes checkpoint subdirectories)
3. For each model, optionally auto-detect K and context from directory name
4. Check which configurations (k, context) haven't been evaluated
5. Submit sbatch jobs only for missing configurations using the shell script
6. Create log files in eval/logs with names matching the model and configuration

Auto-detection patterns:
    K value: Looks for '_K0_', '_K1_', '_K2_', '_K3_', etc. in model directory name
    Context: Looks for '_context_', '_ctx_' (True) or '_nocontext_', '_noctx_' (False)
    Example model name: Qwen2.5-VL-3B-Instruct_all_tasks_revl_1000%100_K0_20251024_133309
        → Auto-detects K=0

Run naming convention:
    {model_name}-{dataset}-k{k_value}[-context]
    Example: Qwen2.5-VL-3B-Instruct_all_tasks_revl_1000%100_K0_20251024_133309-screenspot_bbox_test-k0-context

Log file naming convention:
    {model_name}-{dataset}-k{k_value}[-context].out
"""

import os
import subprocess
import argparse
import wandb
import re
from pathlib import Path


def get_existing_runs(project_name="ReVL_eval", entity=None):
    """
    Get all run names from the wandb project.
    
    Args:
        project_name: Name of the wandb project
        entity: Wandb entity (username or team name). If None, uses default.
    
    Returns:
        Set of run names that already exist
    """
    api = wandb.Api()
    
    # Get runs from the project
    if entity:
        runs = api.runs(f"{entity}/{project_name}")
    else:
        runs = api.runs(project_name)
    
    # Extract run names
    run_names = set()
    for run in runs:
        run_names.add(run.name)
    
    print(f"Found {len(run_names)} existing runs in wandb project '{project_name}'")
    return run_names


def get_model_directories(model_output_dir):
    """
    Get all model directories from the model output folder.
    Excludes checkpoint subdirectories.
    
    Args:
        model_output_dir: Path to the directory containing model subdirectories
    
    Returns:
        List of model directory paths
    """
    model_dirs = []
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(model_output_dir):
        # Skip checkpoint directories
        if 'checkpoint' in os.path.basename(root).lower():
            continue
            
        # Check if this directory contains model files
        # Look for model config files that indicate this is a model directory
        if 'config.json' in files and 'model.safetensors.index.json' in files:
            model_dirs.append(root)
    
    print(f"Found {len(model_dirs)} model directories")
    return model_dirs


def extract_model_name(model_path):
    """
    Extract the model name from the full path.
    Returns the directory name (last component of path).
    """
    return os.path.basename(model_path)


def extract_k_from_model_name(model_name):
    """
    Extract K value from model directory name.
    Looks for patterns like '_K0_', '_K1_', etc.
    
    Args:
        model_name: Name of the model directory
    
    Returns:
        K value as integer, or None if not found
    """
    match = re.search(r'_K(\d+)_', model_name)
    if match:
        return int(match.group(1))
    return None


def extract_context_from_model_name(model_name):
    """
    Extract context setting from model directory name.
    Looks for patterns like '_context_', '_ctx_', '_nocontext_', etc.
    
    Args:
        model_name: Name of the model directory
    
    Returns:
        Boolean indicating context (True/False), or None if not found
    """
    model_name_lower = model_name.lower()
    
    # Look for positive indicators
    if re.search(r'_(with_)?context_', model_name_lower):
        return True
    if re.search(r'_ctx_', model_name_lower):
        return True
    
    # Look for negative indicators
    if re.search(r'_(no|without)_?context_', model_name_lower):
        return False
    if re.search(r'_noctx_', model_name_lower):
        return False
    
    # Not found
    return None


def should_evaluate_model(model_name, existing_runs, dataset, k_values, context_values, auto_detect_k=False, auto_detect_context=False):
    """
    Check if a model should be evaluated.
    
    A model should be evaluated if ANY combination of (dataset, k, context)
    has not been run yet.
    
    Args:
        model_name: Name of the model
        existing_runs: Set of existing run names in wandb
        dataset: Dataset to evaluate on
        k_values: List of k values to try
        context_values: List of context values to try (True/False)
        auto_detect_k: If True, extract K value from model name instead of using k_values
        auto_detect_context: If True, extract context value from model name instead of using context_values
    
    Returns:
        List of tuples (k, context) that need to be evaluated
    """
    missing_configs = []
    
    # If auto_detect_k is enabled, try to extract K from model name
    if auto_detect_k:
        detected_k = extract_k_from_model_name(model_name)
        if detected_k is not None:
            k_values = [detected_k]
            print(f"  Auto-detected K={detected_k} from model name")
        else:
            print(f"  Warning: Could not detect K value from model name, using provided k_values")
    
    # If auto_detect_context is enabled, try to extract context from model name
    if auto_detect_context:
        detected_context = extract_context_from_model_name(model_name)
        if detected_context is not None:
            context_values = [detected_context]
            print(f"  Auto-detected context={detected_context} from model name")
        else:
            print(f"  Warning: Could not detect context value from model name, using provided context_values")
    
    for k in k_values:
        for context in context_values:
            # Construct the expected run name
            run_name = f"{model_name}-{dataset}-k{k}"
            if context:
                run_name += "-context"
            
            if run_name not in existing_runs:
                missing_configs.append((k, context))
    
    return missing_configs


def run_evaluation(model_path, dataset, k, context, eval_script, model_output_dir, base_path, 
                   sbatch_partition, sbatch_time, sbatch_gpus, sbatch_nodes, log_dir):
    """
    Run the evaluation script for a given model and configuration using sbatch.
    
    Args:
        model_path: Full path to the model
        dataset: Dataset to evaluate on
        k: Number of partitions
        context: Whether to keep context
        eval_script: Path to the shell script (e.g., eval-screenspot-test.sh)
        model_output_dir: Base directory for model outputs
        base_path: Base path to use for running the script (typically repo root)
        sbatch_partition: Slurm partition to use
        sbatch_time: Time limit for the job
        sbatch_gpus: GPU specification
        sbatch_nodes: Number of nodes
        log_dir: Directory to store log files
    """
    # Extract model name for display and wandb run naming
    model_name = extract_model_name(model_path)
    
    # Compute relative path from base_path to model_path
    try:
        rel_path = os.path.relpath(model_path, base_path)
    except ValueError:
        # If paths are on different drives (Windows), use absolute path
        rel_path = model_path
    
    # Create log file name based on model, dataset, k, and context
    context_suffix = "-context" if context else ""
    log_filename = f"{model_name}-{dataset}-k{k}{context_suffix}.out"
    log_path = os.path.join(log_dir, log_filename)
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"Full path: {model_path}")
    print(f"Relative path: {rel_path}")
    print(f"Dataset: {dataset}, K: {k}, Context: {context}")
    print(f"Log file: {log_path}")
    print(f"{'='*80}\n")
    
    # Build the sbatch command
    # Convert context to string format expected by shell script
    context_str = "True" if context else "False"
    
    cmd = [
        "sbatch",
        "-p", sbatch_partition,
        "-t", sbatch_time,
        "-N", str(sbatch_nodes),
        "-o", log_path,
        f"--gpus={sbatch_gpus}",
        eval_script,
        "--model", rel_path,
        "--data", dataset,
        "--k", str(k),
        "--context", context_str
    ]
    
    print(f"Submitting job with command:")
    print(f"  {' '.join(cmd)}\n")
    
    try:
        # Submit the job via sbatch
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=base_path  # Run from the project root
        )
        
        # Parse job ID from sbatch output
        job_id = result.stdout.strip()
        print(f"✓ Job submitted successfully: {job_id}")
        print(f"  Model: {model_name}, k={k}, context={context}")
        print(f"  Log: {log_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error submitting job for {model_name} with k={k}, context={context}")
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate models that haven't been evaluated yet"
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/ocean/projects/cis240092p/amartin1/ReVL/finetune/model_output",
        help="Directory containing model subdirectories"
    )
    
    parser.add_argument(
        "--eval_script",
        type=str,
        default="/ocean/projects/cis240092p/amartin1/ReVL/eval/scripts/eval-screenspot-test.sh",
        help="Path to the eval shell script"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="ReVL_eval",
        help="Wandb project name"
    )
    
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Wandb entity (username or team)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="screenspot_bbox_test",
        help="Dataset to evaluate on"
    )
    
    parser.add_argument(
        "--k_values",
        type=str,
        default="0,1,2,3",
        help="Comma-separated list of k values to try (e.g., '0,1,2,3')"
    )
    
    parser.add_argument(
        "--context_values",
        type=str,
        default="true,false",
        help="Comma-separated list of context values (e.g., 'true,false')"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be evaluated without actually running"
    )
    
    parser.add_argument(
        "--base_path",
        type=str,
        default="/ocean/projects/cis240092p/amartin1/ReVL",
        help="Base path for the project (used to compute relative paths)"
    )
    
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/ocean/projects/cis240092p/amartin1/ReVL/eval/logs",
        help="Directory to store log files"
    )
    
    parser.add_argument(
        "--sbatch_partition",
        type=str,
        default="GPU-shared",
        help="Slurm partition to use (e.g., GPU-shared)"
    )
    
    parser.add_argument(
        "--sbatch_time",
        type=str,
        default="48:00:00",
        help="Time limit for each job (e.g., 48:00:00)"
    )
    
    parser.add_argument(
        "--sbatch_gpus",
        type=str,
        default="h100-80:1",
        help="GPU specification (e.g., h100-80:1)"
    )
    
    parser.add_argument(
        "--sbatch_nodes",
        type=int,
        default=1,
        help="Number of nodes to request"
    )
    
    parser.add_argument(
        "--auto_detect_k",
        action="store_true",
        help="Automatically detect K value from model directory name (e.g., '_K0_', '_K1_'). If enabled, ignores --k_values."
    )
    
    parser.add_argument(
        "--auto_detect_context",
        action="store_true",
        help="Automatically detect context setting from model directory name (e.g., '_context_', '_nocontext_'). If enabled, ignores --context_values."
    )
    
    args = parser.parse_args()
    
    # Parse k values and context values
    k_values = [int(k.strip()) for k in args.k_values.split(',')]
    context_values = [v.strip().lower() == 'true' for v in args.context_values.split(',')]
    
    print("Configuration:")
    print(f"  Model directory: {args.model_dir}")
    print(f"  Eval script: {args.eval_script}")
    print(f"  Wandb project: {args.project}")
    print(f"  Dataset: {args.dataset}")
    if args.auto_detect_k:
        print(f"  K values: Auto-detect from model name")
    else:
        print(f"  K values: {k_values}")
    if args.auto_detect_context:
        print(f"  Context values: Auto-detect from model name")
    else:
        print(f"  Context values: {context_values}")
    print(f"  Log directory: {args.log_dir}")
    print(f"  Sbatch partition: {args.sbatch_partition}")
    print(f"  Sbatch time: {args.sbatch_time}")
    print(f"  Sbatch GPUs: {args.sbatch_gpus}")
    print(f"  Sbatch nodes: {args.sbatch_nodes}")
    print(f"  Auto-detect K: {args.auto_detect_k}")
    print(f"  Auto-detect context: {args.auto_detect_context}")
    print(f"  Dry run: {args.dry_run}")
    print()
    
    # Get existing runs from wandb
    existing_runs = get_existing_runs(args.project, args.entity)
    
    # Get all model directories
    model_dirs = get_model_directories(args.model_dir)
    
    if not model_dirs:
        print("No model directories found!")
        return
    
    # Check which models need evaluation
    evaluations_to_run = []
    
    for model_path in model_dirs:
        model_name = extract_model_name(model_path)
        missing_configs = should_evaluate_model(
            model_name, existing_runs, args.dataset, k_values, context_values, args.auto_detect_k, args.auto_detect_context
        )
        
        if missing_configs:
            print(f"\nModel '{model_name}' needs evaluation:")
            for k, context in missing_configs:
                print(f"  - k={k}, context={context}")
                evaluations_to_run.append((model_path, k, context))
        else:
            print(f"\nModel '{model_name}' - all configurations already evaluated ✓")
    
    print(f"\n{'='*80}")
    print(f"Total evaluations to run: {len(evaluations_to_run)}")
    print(f"{'='*80}\n")
    
    if args.dry_run:
        print("Dry run mode - not executing evaluations")
        return
    
    if not evaluations_to_run:
        print("No evaluations needed!")
        return
    
    # Run evaluations
    successful = 0
    failed = 0
    
    for i, (model_path, k, context) in enumerate(evaluations_to_run, 1):
        print(f"\n[{i}/{len(evaluations_to_run)}]")
        success = run_evaluation(
            model_path, args.dataset, k, context, args.eval_script, args.model_dir, args.base_path,
            args.sbatch_partition, args.sbatch_time, args.sbatch_gpus, args.sbatch_nodes, args.log_dir
        )
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total: {len(evaluations_to_run)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

