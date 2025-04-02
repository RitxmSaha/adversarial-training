#!/usr/bin/env python3
from datasets import load_dataset
import subprocess
import json
import os
from pathlib import Path
from evalplus.data import get_human_eval_plus, write_jsonl

def execute_tests(sample_file):
    """
    Execute evalplus evaluation on a sample file and return the results.
    
    Args:
        sample_file (str): Path to the sample JSONL file to evaluate
        
    Returns:
        dict: The evaluation results parsed from the JSON file
    """

    file_path = Path(sample_file)
    base_name = file_path.stem
    result_file = f"{base_name}.eval_results.json"

    cmd = [
        "docker", "exec", "evalplus-container", "evalplus.evaluate",
        "--dataset", "humaneval",
        "--samples", f"/host_code/{file_path.name}",
        "--test_details", 
        "--parallel", "1"
    ]

    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Evaluation completed: {result.stdout.decode()}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing evalplus: {e}")
        print(f"Stderr: {e.stderr.decode()}")
        return None
    
    # Read the results file
    result_path = Path(file_path.parent) / result_file
    try:
        results = []
        with open(result_path, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    results.append(json.loads(line))
        return results[0] if results else None  # Return first result or None
    except FileNotFoundError:
        print(f"Results file not found: {result_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error parsing results file: {result_path}")
        return None

def main():
    # Load HumanEvalPlus dataset
    human_eval_plus = get_human_eval_plus()
    
    # Create directory for output files using pathlib
    output_dir = Path("host_code")
    output_dir.mkdir(exist_ok=True)
    
    # Process first 10 HumanEval cases using slicing
    samples = [
        {
            "task_id": task_id,
            "completion": problem["canonical_solution"]
        }
        for task_id, problem in list(human_eval_plus.items())[:10]
    ]

    for i, sample in enumerate(samples):
        output_file = output_dir / f"HumanEval_{i}.jsonl"
        write_jsonl(output_file, [sample])
    
    print(f"Generated {len(samples)} JSONL files in {output_dir}")

    #evaluate the samples
    """
    docker exec evalplus-container evalplus.evaluate \
        --dataset humaneval \
        --samples /host_code/HumanEval_0.jsonl
    """

if __name__ == "__main__":
    #main()
    output = execute_tests("host_code/HumanEval_3.jsonl")
    print("#"*30)
    print(json.dumps(output, indent=4))
