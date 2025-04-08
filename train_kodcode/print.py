import os
import pandas as pd

# Path to the saved parquet file
train_parquet_path = "rl_data/train.parquet"  # or wherever your file is saved

# Load the data
if os.path.exists(train_parquet_path):
    df = pd.read_parquet(train_parquet_path)
    
    # Get the first example
    if len(df) > 0:
        example = df.iloc[0]
        
        print("=== SAMPLE PROCESSED LEETCODE ENTRY ===")
        print("\n--- Basic Information ---")
        print(f"Data Source: {example['data_source']}")
        print(f"Ability: {example['ability']}")
        
        print("\n--- Prompt ---")
        for msg in example['prompt']:
            print(f"Role: {msg['role']}")
            print(f"Content (first 300 chars): {msg['content']}...")
        
        print("\n--- Reward Model ---")
        print(f"Style: {example['reward_model']['style']}")
        print(f"Ground Truth (first 300 chars): {example['reward_model']['ground_truth']}...")
        
        print("\n--- Extra Info ---")
        print(f"Split: {example['extra_info']['split']}")
        print(f"Task ID: {example['extra_info']['task_id']}")
        
        if 'metadata' in example['extra_info']:
            print("\n--- Metadata ---")
            for key, value in example['extra_info']['metadata'].items():
                if isinstance(value, str):
                    print(f"{key}: {value}..." if len(value) > 100 else f"{key}: {value}")
                else:
                    print(f"{key}: {value}")
    else:
        print("The DataFrame is empty.")
else:
    print(f"File not found: {train_parquet_path}")
