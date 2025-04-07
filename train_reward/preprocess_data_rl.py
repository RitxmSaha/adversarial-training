import re
import os
import json
import pandas as pd
from evalplus.data import get_human_eval_plus
from verl.utils.hdfs_io import copy, makedirs
import argparse

def extract_solution(solution_str):
    """Extract the solution from the code solution string."""
    pass

def format_prompt_with_test_cases(original_prompt, proxy_test_cases):
    """
    Replaces examples in the original prompt with test cases.
    """
    # Find where the examples start in the original prompt
    lines = original_prompt.strip().split('\n')
    example_index = -1
    indent = "    "  # Default indent
    
    # Extract function name from the signature
    function_name = None
    for line in lines:
        if line.strip().startswith('def '):
            # Extract function name from "def function_name(params):"
            function_signature = line.strip()
            function_name = function_signature.split('def ')[1].split('(')[0].strip()
            break
    
    # If function name not found, default to original behavior
    if not function_name:
        return original_prompt
    
    # Find the first example line and its indentation
    for i, line in enumerate(lines):
        if '>>>' in line:
            example_index = i
            indent = line[:line.find('>>>')]
            break
    
    # If no examples found, return original prompt
    if example_index == -1:
        return original_prompt
    
    # Keep everything before the examples
    result_lines = lines[:example_index]
    
    # Add evaluation statement before examples
    evaluation_statement = f"\n{indent}The following examples will be used for evaluation:"
    result_lines.append(evaluation_statement)
    
    # Extract test cases from proxy_test_cases
    test_cases = []
    in_check_function = False
    
    for line in proxy_test_cases.split('\n'):
        if line.strip().startswith('def check(candidate):'):
            in_check_function = True
        elif in_check_function and line.strip().startswith('assert candidate('):
            test_cases.append(line.strip())
    
    # Format test cases as examples with proper indentation
    for tc in test_cases:
        tc = tc.replace('assert ', '')
        if ' == ' in tc:
            # Replace "candidate" with the actual function name
            call, expected = tc.split(' == ')
            call = call.replace('candidate', function_name)
            result_lines.append(f"{indent}>>> {call}")
            result_lines.append(f"{indent}{expected}")
    
    # Find where docstring ends to add remaining content
    docstring_end = -1
    for i in range(example_index, len(lines)):
        if '"""' in lines[i] or "'''" in lines[i]:
            docstring_end = i
            break
    
    # Add closing docstring and any content after it
    if docstring_end != -1:
        result_lines.append(lines[docstring_end])
        result_lines.extend(lines[docstring_end+1:])
    else:
        result_lines.append(f'{indent}"""')
    
    return '\n'.join(result_lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/opt/tiger/evalplus')
    parser.add_argument('--hdfs_dir', default=None)
    
    args = parser.parse_args()
    
    data_source = 'evalplus'
    
    human_eval_plus = get_human_eval_plus()
    
    # Convert the dictionary format to a list for easier processing
    dataset_list = []
    for task_id, problem in human_eval_plus.items():
        problem['task_id'] = task_id
        dataset_list.append(problem)
    
    # For simplicity, we'll use 80% of data for training and 20% for testing
    # You can adjust this split or use a different strategy
    split_idx = int(len(dataset_list) * 0.8)
    train_data = dataset_list[10:split_idx]  # Skip first 10 samples
    test_data = dataset_list[split_idx:]
    
    def make_map_fn(split):
        def process_fn(problem, idx):
            prompt = problem.get('prompt', '')
            proxy_test_cases = problem.get('test', '')
            solution = problem.get('canonical_solution', '')

            reformulated_prompt = format_prompt_with_test_cases(prompt, proxy_test_cases)

            instruction = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\nThe assistant first thinks about the reasoning process in the mind and then provides the user\nwith the answer. The reasoning process and answer are enclosed within <think> <\/think> and\n<answer> <\/answer> tags, respectively, i.e., <think> reasoning process here <\/think>\n<answer> answer here <\/answer>.\n\n User: Implement the following function:\n\n{reformulated_prompt}\n\n Assistant:"            
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": instruction
                }],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution,
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'task_id': problem.get('task_id', f'task_{idx}'),
                    'prompt': instruction,
                }
            }
            return data
        
        return process_fn
    
    # Process the train and test dataL
    train_processed = [make_map_fn('train')(example, idx) for idx, example in enumerate(train_data)]
    test_processed = [make_map_fn('test')(example, idx) for idx, example in enumerate(test_data)]
    
    # Convert to pandas DataFrame and then to parquet
    train_df = pd.DataFrame(train_processed)
    test_df = pd.DataFrame(test_processed)
    
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)
    
    # Save the datasets to parquet files
    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_df.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    # Copy to HDFS if specified
    if hdfs_dir:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
