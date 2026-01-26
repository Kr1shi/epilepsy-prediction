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
    python data_segmentation.py

    echo "Running data preprocessing..."
    python data_preprocessing.py

    echo "Running model training..."
    python train.py
    
    echo "Evaluating..."
    python evaluate_train.pys

    echo "----------------------------------------"
    echo "Pipeline finished at $(date)"
} 2>&1 | tee "$LOG_FILE"
