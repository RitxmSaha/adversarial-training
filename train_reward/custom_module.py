import re
import subprocess
import tempfile
import os
import json
import time
import csv
from pathlib import Path
import uuid
from evalplus.data import write_jsonl
from typing import Dict, Any, List, Optional

# Try to import filelock for proper inter-process locking
try:
    from filelock import FileLock
    has_filelock = True
except ImportError:
    print("Warning: filelock package not found. File locking will be unreliable.")
    has_filelock = False
    import threading
    file_lock = threading.Lock()

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[Dict[str, Any]] = None) -> float:
    """
    Compute reward score for code generation tasks.
    
    For evalplus dataset, we'll use a combination of:
    1. Syntax correctness
    3. Test case passing (simulated)
    
    Args:
        data_source: The name of the dataset
        solution_str: The generated solution
        ground_truth: The reference/ground truth solution
        extra_info: Additional information about the task
        
    Returns:
        float: Reward score between 0 and 1
    """
    task_id = extra_info.get('task_id', '') if extra_info else ''
    
    def get_answer_string(text):
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    answer_string = get_answer_string(solution_str)
    
    base_score, plus_score = test_reward(task_id, answer_string)
    format_score = format_reward(solution_str)
    length_penalty = -max(0, (len(solution_str) - 2048) / (2048*4))
    final_score = base_score*0.6 + format_score*0.4 + length_penalty
    
    # Extract training metadata from extra_info
    epoch = extra_info.get('epoch', 'unknown') if extra_info else 'unknown'
    step = extra_info.get('step', 'unknown') if extra_info else 'unknown'
    is_validation = extra_info.get('is_validation', False) if extra_info else False
    prompt = extra_info.get('prompt', data_source) if extra_info else data_source
    
    # Log all the details to file
    metrics = {
        "task_id": task_id,
        "epoch": epoch,
        "step": step,
        "is_validation": is_validation,
        "base_score": base_score,
        "plus_score": plus_score,
        "format_score": format_score,
        "length_penalty": length_penalty,
        "final_score": final_score,
        "prompt": data_source,
        "completion": solution_str,
    }
    
    log_metrics(metrics)
    
    return final_score


def log_metrics(metrics: Dict[str, Any]) -> None:
    """
    Log metrics to JSONL files.
    
    Args:
        metrics: Dictionary containing all metrics and data to log
    """
    # Create a completions directory
    completions_dir = os.path.join(os.getcwd(), "completions")
    os.makedirs(completions_dir, exist_ok=True)
    
    # Create a step directory
    step_dir = os.path.join(completions_dir, f"step_{metrics['step']}")
    os.makedirs(step_dir, exist_ok=True)
    
    # Determine which JSONL file to use
    jsonl_filename = "validate.jsonl" if metrics['is_validation'] else "train.jsonl"
    jsonl_path = os.path.join(step_dir, jsonl_filename)
    
    # Create a lock file path
    lock_file = f"{jsonl_path}.lock"
    
    # Prepare row data - truncate long strings
    row_data = {
        "task_id": metrics['task_id'],
        "epoch": metrics['epoch'],
        "step": metrics['step'],
        "is_validation": metrics['is_validation'],
        "prompt": metrics['prompt'][:1000],  # Truncate very long prompts
        "completion": metrics['completion'][:1000],  # Truncate very long completions
        "base_score": metrics['base_score'],
        "plus_score": metrics['plus_score'],
        "format_score": metrics['format_score'],
        "length_penalty": metrics['length_penalty'],
        "final_score": metrics['final_score'],
        "timestamp": time.time()
    }
    
    # Write to JSONL file with proper locking
    if has_filelock:
        with FileLock(lock_file):
            with open(jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(row_data, ensure_ascii=False) + '\n')
    else:
        # Fallback to threading.Lock (won't work across processes)
        with file_lock:
            with open(jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(row_data, ensure_ascii=False) + '\n')


def format_reward(solution):
    """
    Checks if the solution follows the required format:

    <think>...</think>
    <answer>...</answer>

    Args:
        solution: The generated solution
        
    Returns:
        float: Reward score between 0 and 1 based on format compliance.
    """
    cleaned = solution.strip()
    
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, cleaned, re.DOTALL)
    
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, cleaned, re.DOTALL)
    
    # Ensure both sections exist
    if not think_match or not answer_match:
        return 0.0
    
    cleaned = cleaned.replace(think_match.group(0), '')
    cleaned = cleaned.replace(answer_match.group(0), '')
    cleaned = cleaned.strip()
    
    # Format reward is 1 if there's no text outside the tags, 0 otherwise
    return 1.0 if not cleaned else 0.0

def test_reward(task_id, solution):
    """
    Compute test reward for code generation tasks.
    
    Args:
        task_id: The ID of the task
        solution: The generated solution
        
    Returns:
        float: Reward score between 0 and 1
    """
    results = evaluate_solution(task_id, solution)
    if results:
        return results["pass_at_k"]["base"]["pass@1"], results["pass_at_k"]["plus"]["pass@1"]
    else:
        return 0.0, 0.0

def evaluate_solution(task_id, solution):
    """
    Evaluate a solution for a specific HumanEval task using EvalPlus.
    Runs only the base tests.
    
    Args:
        task_id (str): The HumanEval task ID (e.g., "HumanEval/3")
        solution (str): The solution code as a string
        
    Returns:
        dict: The evaluation results
    """
    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent.parent / "results"
    
    # Generate a random UUID for the filename
    unique_id = str(uuid.uuid4())
    file_name = f"{unique_id}.jsonl"
    sample_file = results_dir / file_name
    
    # Create a sample with the given task_id and solution
    sample = {
        "task_id": task_id,
        "solution": solution
    }
    
    # Write the sample to a file
    write_jsonl(sample_file, [sample])
    
    # Run the evaluation with base tests only
    cmd = [
        "docker", "exec", "evalplus-container", "evalplus.evaluate",
        "--dataset", "humaneval",
        "--samples", f"/results/{file_name}",
        "--test_details",
        "--parallel", "8"
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Evaluation completed for {task_id}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing evalplus: {e}")
        print(f"Stderr: {e.stderr.decode()}")
        return None
    
    # Read the results file
    result_file = f"{sample_file.stem}.eval_results.json"
    result_path = results_dir / result_file
    
    try:
        results = []
        with open(result_path, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    results.append(json.loads(line))
        
        # Store the results before deleting files
        result_data = results[0] if results else None
        
        # Delete the results file and the original sample file
        if result_path.exists():
            result_path.unlink()
            print(f"Deleted results file: {result_path}")
        
        if sample_file.exists():
            sample_file.unlink()
            print(f"Deleted sample file: {sample_file}")
        
        return result_data
    except FileNotFoundError:
        print(f"Results file not found: {result_path}")
        # Delete the sample file even if results file is not found
        if sample_file.exists():
            sample_file.unlink()
            print(f"Deleted sample file: {sample_file}")
        return None
    except json.JSONDecodeError:
        print(f"Error parsing results file: {result_path}")
        # Delete both files in case of JSON decode error
        if result_path.exists():
            result_path.unlink()
            print(f"Deleted results file: {result_path}")
        
        if sample_file.exists():
            sample_file.unlink()
            print(f"Deleted sample file: {sample_file}")
        return None


    
    
    
    
