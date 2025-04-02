#!/bin/bash

# Set output directory to current directory
OUTPUT_DIR="$(pwd)"

# Run the preprocessing script
python preprocess_data.py --local_dir "$OUTPUT_DIR"

# Check if the preprocessing was successful
if [ -f "$OUTPUT_DIR/train.parquet" ] && [ -f "$OUTPUT_DIR/test.parquet" ]; then
    echo "✅ Preprocessing complete!"
    echo "Output files:"
    echo "- $OUTPUT_DIR/train.parquet"
    echo "- $OUTPUT_DIR/test.parquet"
else
    echo "❌ Preprocessing failed. Check for errors above."
fi