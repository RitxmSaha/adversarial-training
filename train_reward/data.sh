#!/bin/bash

# Set output directory to current directory
OUTPUT_DIR_RL="$(pwd)/rl_data"
OUTPUT_DIR_SFT="$(pwd)/sft_data"

# Run the preprocessing script
python preprocess_data_rl.py --local_dir "$OUTPUT_DIR_RL"
python preprocess_data_sft.py --json_path ./sft_prompts.json --output_dir "$OUTPUT_DIR_SFT"

# Check if the preprocessing was successful
if [ -f "$OUTPUT_DIR_RL/train.parquet" ] && [ -f "$OUTPUT_DIR_RL/test.parquet" ]; then
    echo $'\n✅ Preprocessing complete for RL data!'
    echo "Output files:"
    echo "- $OUTPUT_DIR_RL/train.parquet"
    echo "- $OUTPUT_DIR_RL/test.parquet"
else
    echo "❌ Preprocessing for RL data failed. Check for errors above."
fi

# Check if the SFT preprocessing was successful
if [ -f "$OUTPUT_DIR_SFT/train.parquet" ] && [ -f "$OUTPUT_DIR_SFT/test.parquet" ]; then
    echo $'\n✅ Preprocessing complete for SFT data!'
    echo "Output files:"
    echo "- $OUTPUT_DIR_SFT/train.parquet"
    echo "- $OUTPUT_DIR_SFT/test.parquet"
else
    echo "❌ Preprocessing for SFT data failed. Check for errors above."
fi