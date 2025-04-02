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
    train_data = dataset_list[:split_idx]
    test_data = dataset_list[split_idx:]
    
    def make_map_fn(split):
        def process_fn(problem, idx):
            prompt = problem.get('prompt', '')
            proxy_test_cases = problem.get('test', '')
            solution = problem.get('canonical_solution', '')
            
            # Update the instruction format to include <think> and <answer> tags
            instruction = f"Implement the following function:\n\n{prompt}\n\nThe following testcases provided below are run afterwards to determine if your implementation is correct{proxy_test_cases}\n\nFirst think through the problem step by step in the <think> </think> section, then provide your final implementation in the <answer> </answer> section.\n\n Example:\n\n<think>reason about problem here<think><answer> final formatted answer here<answer>"
            
            
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
                    'task_id': problem.get('task_id', f'task_{idx}')
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
