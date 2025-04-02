#!/usr/bin/env python3
import subprocess
import json
from pathlib import Path
from evalplus.data import write_jsonl

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
    
    # Format the filename
    file_name = f"{task_id.replace('/', '_')}.jsonl"
    sample_file = results_dir / file_name
    
    # Create a sample with the given task_id and solution
    sample = {
        "task_id": task_id,
        "completion": solution
    }
    
    # Write the sample to a file
    write_jsonl(sample_file, [sample])
    
    # Run the evaluation with base tests only
    cmd = [
        "docker", "exec", "evalplus-container", "evalplus.evaluate",
        "--dataset", "humaneval",
        "--samples", f"/results/{file_name}",
        "--test_details",
        "--base_only",
        "--parallel", "1"
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
        return results[0] if results else None
    except FileNotFoundError:
        print(f"Results file not found: {result_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error parsing results file: {result_path}")
        return None
