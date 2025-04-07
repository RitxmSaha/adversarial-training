import pandas as pd
import json
import os
import argparse
import numpy as np

def convert_json_to_parquet(json_path, output_dir, test_size=0.2, random_state=42):
    """
    Convert JSON data into parquet files for training and testing.
    
    Args:
        json_path: Path to the JSON file
        output_dir: Directory to save the parquet files
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract data with the correct format for SFTDataset
    # We need to create Series objects that will be compatible with the tolist() method
    prompts = pd.Series([item['prompt'] for item in data])
    responses = pd.Series([item['response'] for item in data])
    
    # For simplicity, we'll use 80% of data for training and 20% for testing
    split_idx = int(len(prompts) * 0.8)
    
    # Create DataFrames with explicit series
    train_df = pd.DataFrame({
        'prompt': prompts[:split_idx], 
        'response': responses[:split_idx]
    })
    test_df = pd.DataFrame({
        'prompt': prompts[split_idx:], 
        'response': responses[split_idx:]
    })
    
    # Verify the DataFrame structure
    print(f"Training DataFrame columns: {train_df.columns}")
    print(f"First row prompt type: {type(train_df['prompt'].iloc[0])}")
    print(f"First row prompt value (truncated): {train_df['prompt'].iloc[0][:100]}...")
    
    # Save as parquet files
    train_path = os.path.join(output_dir, 'train.parquet')
    test_path = os.path.join(output_dir, 'test.parquet')
    
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    
    print(f"Created train parquet file with {len(train_df)} examples at: {train_path}")
    print(f"Created test parquet file with {len(test_df)} examples at: {test_path}")
    
    return train_path, test_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON to parquet files for training and testing")
    parser.add_argument('--json_path', default='./sft_prompts.json', help='Path to the JSON file')
    parser.add_argument('--output_dir', default='./sft_data', help='Directory to save the parquet files')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset for testing')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    train_path, test_path = convert_json_to_parquet(
        args.json_path, 
        args.output_dir,
        args.test_size,
        args.random_state
    )
