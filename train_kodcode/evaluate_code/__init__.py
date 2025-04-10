import re
from typing import Tuple, Optional
import time
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

import numpy as np

from .utils import _ERROR_MSG_PREFIX

# Try to import filelock for proper inter-process locking
try:
    from filelock import FileLock
    has_filelock = True
except ImportError:
    print("Warning: filelock package not found. File locking will be unreliable.")
    has_filelock = False
    import threading
    file_lock = threading.Lock()

_MAX_CHAR_DISPLAY = 2048
_DEBUG = False
CODER1_EXEC = os.environ.get("CODER1_EXEC", "firejail")

if CODER1_EXEC == "docker":
    from .docker_exec import code_exec_docker
    code_exec = code_exec_docker
elif CODER1_EXEC == "firejail":
    from .firejail_exec import code_exec_firejail
    code_exec = code_exec_firejail
elif CODER1_EXEC == "ces":
    from .ces_exec import remote_code_exec_ces
    code_exec = remote_code_exec_ces
elif CODER1_EXEC == "kira":
    from .kira_exec import remote_code_exec_kira
    code_exec = remote_code_exec_kira
else:
    raise ValueError(f"Unknown CODER1_EXEC: {CODER1_EXEC}")


def remote_check_stdio(code, stdin, stdout):
    succ, output = code_exec(code=code, stdin=stdin)
    return succ, output, stdin, stdout



# https://github.com/Unakar/Logic-RL/blob/main/verl/utils/reward_score/kk.py
def try_extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if matches:
        final_answer = matches[-1].group(1).strip()
        return final_answer

    return solution_str


CODE_PATTERN = re.compile(r'```(?:\w+)?\n(.*?)\n```', re.DOTALL)


def extract_code_from_string(solution_str):
    solution_str = try_extract_solution(solution_str)
    code_blocks = CODE_PATTERN.findall(solution_str)
    return '\n'.join(code_blocks).strip()


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
        float: Reward score
    """
    format_reward = float(calculate_format(solution_str.strip()))
    correct_answer_base, correct_answer_plus, logs = calculate_correctness(solution_str.strip(), ground_truth, extra_info)

    reward = 0.1 *  format_reward + 0.9 * correct_answer_base

    # Log all the details to file
    metrics = {
        "data_source": data_source,
        "epoch": extra_info["epoch"],
        "step": extra_info["step"],
        "is_validation": extra_info["is_validation"],
        "correct_score_base": float(correct_answer_base),
        "correct_score_plus": float(correct_answer_plus),
        "format_score": format_reward,
        "final_score": reward,
        "prompt": extra_info["prompt"],
        "completion": solution_str,
    }
    
    log_metrics(metrics)

    marker = "✅" if 1.0 == reward else "❌"

    if _DEBUG:
        print(f"=" * 60)
        reward_log = "Reward Summary " + marker * 1 + "\nReward Log:" + logs + "\n" + f"Final Reward = {reward} " + marker * 1
        print(reward_log + "\n")
        print(f"=" * 60)
    else:
        reward_log = f"Reward Summary {marker} /// Final Reward = {reward}"
        print(reward_log)
    return reward

def calculate_format(processed_str: str) -> float:
    """
    Calculate format score with partial credit for different format elements.
    Return values:
    - 1.0: Perfect format (has think/answer tags and code blocks)
    - 0.5: Partial format (has answer tags but missing code blocks)
    - 0.3: Basic format (has answer tags but incorrect formatting)
    - 0.5: Partial format (has think tags but missing answer tags)
    - -1.0: Missing essential elements
    """
    # Initialize score
    score = -1.0
    
    # Check for the basic think/answer structure
    think_pattern = re.compile(r'<think>.*</think>', re.DOTALL)
    has_think = bool(think_pattern.search(processed_str.strip()))
    
    # Check for answer tags
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    answer_match = answer_pattern.search(processed_str.strip())
    
    if answer_match:
        # Has answer tags, start with base score
        score = 0.3
        
        # Check if code is properly wrapped in markdown code blocks
        answer_content = answer_match.group(1).strip()
        code_block_pattern = re.compile(r'```(?:python)?\n.*?\n```', re.DOTALL)
        code_block_in_answer = bool(code_block_pattern.search(answer_content))
        
        # Add points for having code blocks
        if code_block_in_answer:
            score += 0.7  # Makes it 1.0 total
        
        # Add points for having think tags as well
        if has_think:
            # Don't exceed 1.0 total
            score = min(1.0, score + 0.2)
    elif has_think:
        # Only has think tags, no answer tags
        score = -0.5  # Better than nothing, but still invalid
    
    return score

def calculate_correctness(solution_str: str, ground_truth: str, extra_info: Optional[Dict[str, Any]] = None) -> float:
    pass_fmt = calculate_format(solution_str)
    solution_code = extract_code_from_string(solution_str)
    test_code = json.loads(extra_info["tests"])["base_tests"]
    test_code_plus = json.loads(extra_info["tests"])["plus_tests"]
    logs = []

    if not pass_fmt == 1 or len(solution_code) == 0:
        logs.append("Bad format detected!")
        logs.append("Original Model Output:")
        logs.append("-" * 32)
        logs.append(solution_str)
        logs.append("-" * 32)
        return -1, -1, "\n".join(logs)

    
    t_start = time.time()
    succ, err = code_exec(solution_code, pytest=test_code)
    succ_plus, err_plus = code_exec(solution_code, pytest=test_code_plus)

    if not succ or not succ_plus:
        logs.append("!" * 16 + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s" + "!" * 16)
        logs.append(err[:_MAX_CHAR_DISPLAY])
        logs.append("-" * 16 + "Failed Prompt" + "-" * 16)
        logs.append(extra_info["prompt"].replace("\n\n", "\n"))
        
        return succ, succ_plus, "\n".join(logs)

    return succ, succ_plus, "\n".join(logs)

def log_metrics(metrics: Dict[str, Any]) -> None:
    """
    Log metrics to JSONL files.
    
    Args:
        metrics: Dictionary containing all metrics and data to log
    """
    # Create a completions directory
    completions_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(completions_dir, exist_ok=True)
    
    # Find the most recent date-time directory
    folders = [f for f in os.listdir(completions_dir) if os.path.isdir(os.path.join(completions_dir, f))]
    
    # Filter for date-time formatted folders (YYYY-MM-DD-HH-MM-SS)
    datetime_folders = [f for f in folders if re.match(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', f)]
    
    if datetime_folders:
        # Sort by date-time (most recent last)
        sorted_folders = sorted(datetime_folders, 
                               key=lambda x: time.mktime(time.strptime(x, "%Y-%m-%d-%H-%M-%S")))
        most_recent_dir = os.path.join(completions_dir, sorted_folders[-1])
    else:
        # Throw an exception if no date-time directories exist
        raise FileNotFoundError("No date-time directories found in logs folder. Please create a directory with format DD-MM-YYYY-HH-MM first.")
    
    # Create a step directory inside the most recent date-time directory
    step_dir = os.path.join(most_recent_dir, f"step_{metrics['step']}")
    os.makedirs(step_dir, exist_ok=True)
    
    # Determine which JSONL file to use
    jsonl_filename = "validate.jsonl" if metrics['is_validation'] else "train.jsonl"
    jsonl_path = os.path.join(step_dir, jsonl_filename)
    
    # Create a lock file path
    lock_file = f"{jsonl_path}.lock"
    
    row_data = {
        "data_source": metrics['data_source'],
        "epoch": metrics['epoch'],
        "step": metrics['step'],
        "is_validation": metrics['is_validation'],
        "correct_score_base": metrics['correct_score_base'],
        "correct_score_plus": metrics['correct_score_plus'],
        "format_score": metrics['format_score'],
        "final_score": metrics['final_score'],
        "timestamp": time.time(),
        "prompt": metrics['prompt'],
        "completion": metrics['completion']
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