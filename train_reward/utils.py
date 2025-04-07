#!/usr/bin/env python3
import subprocess
import json
from pathlib import Path
from evalplus.data import write_jsonl
import uuid  # Add this import at the top with other imports

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
