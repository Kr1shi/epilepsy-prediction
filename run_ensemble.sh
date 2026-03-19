#!/bin/bash

caffeinate -dims &
CAFFEINE_PID=$!
echo "Caffeinate started (PID: $CAFFEINE_PID)"

{
    echo "Starting ensemble pipeline at $(date)"
    echo "----------------------------------------"

    echo "Training ensemble (5 seeds: pretrain + fine-tune)..."
    python3 train.py --ensemble 5

    echo "Evaluating ensemble..."
    python3 evaluate_test.py --ensemble 5

    echo "----------------------------------------"
    echo "Ensemble pipeline finished at $(date)"
} 2>&1 | tee ensemble_log.txt

kill $CAFFEINE_PID 2>/dev/null
echo "Done. Caffeinate stopped."
