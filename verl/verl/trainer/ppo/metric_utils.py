# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics related to the PPO trainer.
"""

import torch
import os
import re
import json
import time
from typing import Any, Dict, List, Optional
from glob import glob
import numpy as np
from verl import DataProto


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info['global_token_num'])
    time = timing_raw['step']
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        'perf/total_num_tokens': total_num_tokens,
        'perf/time_per_step': time,
        'perf/throughput': total_num_tokens / (time * n_gpus),
    }


def load_all_recent_metrics(log_type: str = "train") -> List[Dict[str, Any]]:
    """
    Loads all metrics from the most recent log file.
    
    Args:
        log_type: Either "train" or "validate" to specify which log file to read
    
    Returns:
        List of dictionaries containing all metrics entries from the most recent log file
    """
    # Find the most recent datetime folder
    logs_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(logs_dir):
        return []
        
    # Get all datetime folders in YYYY-MM-DD-HH-MM-SS format
    datetime_folders = [
        d for d in os.listdir(logs_dir) 
        if os.path.isdir(os.path.join(logs_dir, d)) and 
        re.match(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', d)
    ]
    
    if not datetime_folders:
        return []
        
    # Sort to get the most recent folder
    sorted_folders = sorted(
        datetime_folders,
        key=lambda x: time.mktime(time.strptime(x, "%Y-%m-%d-%H-%M-%S"))
    )
    most_recent_folder = os.path.join(logs_dir, sorted_folders[-1])
    
    # Find all step folders within the most recent datetime folder
    step_folders = glob(os.path.join(most_recent_folder, "step_*"))
    if not step_folders:
        return []
        
    # Sort step folders by step number
    sorted_step_folders = sorted(
        step_folders,
        key=lambda x: int(os.path.basename(x).split("_")[1])
    )
    most_recent_step_folder = sorted_step_folders[-1]
    
    # Path to the requested jsonl file (train or validate)
    jsonl_path = os.path.join(most_recent_step_folder, f"{log_type}.jsonl")
    if not os.path.exists(jsonl_path):
        return []
    
    # Read all entries from the jsonl file
    try:
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                return []
            
            # Parse all json entries
            all_metrics = []
            for line in lines:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        all_metrics.append(entry)
                    except json.JSONDecodeError:
                        continue
                        
            return all_metrics
    except Exception as e:
        print(f"Error reading metrics file {jsonl_path}: {e}")
        return []


def compute_recent_train_metrics(batch: Optional[DataProto] = None, timing_raw: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Reports score metrics from both train and validation most recent log files.
    Calculates mean, max, and min values for format_score, final_score, and correct_score.
    
    Args:
        batch: Optional DataProto object (not used, for API consistency)
        
    Returns:
        Dictionary containing aggregated metrics statistics
    """
    train_entries = load_all_recent_metrics("train")
    validate_entries = load_all_recent_metrics("validate")
    
    metrics = {}
    
    # Define fields to calculate statistics for
    score_fields = ['format_score', 'final_score', 'correct_score_base', 'correct_score_plus']
    
    # Process validate metrics
    if validate_entries:
        for field in score_fields:
            # Extract all values for this field (skipping non-numeric and missing values)
            values = [entry[field] for entry in validate_entries 
                     if field in entry and isinstance(entry[field], (int, float))]
            
            if values:
                metrics[f'recent_validation/{field}/mean'] = np.mean(values)
    
    # Process train metrics
    if train_entries:
        for field in score_fields:
            # Extract all values for this field (skipping non-numeric and missing values)
            values = [entry[field] for entry in train_entries 
                     if field in entry and isinstance(entry[field], (int, float))]
            
            if values:
                metrics[f'recent_train/{field}/mean'] = np.mean(values)
    
    return metrics
