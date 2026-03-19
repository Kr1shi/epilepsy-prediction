#!/bin/bash

caffeinate -dims &
CAFFEINE_PID=$!
echo "Caffeinate started (PID: $CAFFEINE_PID)"

{
    echo "Starting pipeline at $(date)"
    echo "----------------------------------------"

    echo "Running pretraining (FocalLoss)..."
    python3 train.py --pretrain

    echo "Running per-patient fine-tuning..."
    python3 train.py

    echo "Running evaluation..."
    python3 evaluate_test.py

    echo "----------------------------------------"
    echo "Pipeline finished at $(date)"
} 2>&1 | tee pipeline_log.txt

kill $CAFFEINE_PID 2>/dev/null
echo "Done. Caffeinate stopped."
