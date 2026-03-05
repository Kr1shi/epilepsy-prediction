#!/bin/bash

# Define log file
LOG_FILE="pipeline_log.txt"

# Run the pipeline and redirect both stdout and stderr to the log file
{
    echo "Starting pipeline at $(date)"
    echo "----------------------------------------"

    echo "Cleaning up previous models and preprocessed data..."
    rm -rf model/
    rm -rf preprocessing/

    echo "Running data segmentation..."
    python3 data_segmentation.py

    echo "Running data preprocessing..."
    python3 data_preprocessing.py

    echo "Running model training..."
    python3 train.py > output.txt
    
    echo "Evaluating..."
    python3 evaluate_test.py

    echo "----------------------------------------"
    echo "Pipeline finished at $(date)"
} 2>&1 | tee "$LOG_FILE"
