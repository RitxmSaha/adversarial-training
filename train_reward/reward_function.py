import re
import subprocess
import tempfile
import os
import json
from typing import Dict, Any, List, Optional
from utils import evaluate_solution

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

    print(f"[EVALUATION LOG] Task ID: {task_id}")
    print(f"[EVALUATION LOG] Solution being evaluated:")
    print("="*50)
    print(solution_str)
    print("="*50)
    
    def get_answer_string(text):
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        return match.group(1).strip() if match else ""
    

    
    answer_string = get_answer_string(solution_str)
    
    test_score = test_reward(task_id, answer_string)
    format_score = format_reward(solution_str)
    final_score = test_score*0.5 + format_score*0.5
    
    return final_score


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
        return results["pass_at_k"]["base"]["pass@1"]
    else:
        return 0.0


    
    
    
    
