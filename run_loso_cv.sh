#!/bin/bash

caffeinate -dims &
CAFFEINE_PID=$!
echo "Caffeinate started (PID: $CAFFEINE_PID)"

{
    echo "Starting LOSO-CV pipeline at $(date)"
    echo "----------------------------------------"

    echo "Step 1: Creating master HDF5 datasets (preprocessing)..."
    python3 data_preprocessing.py --master

    echo "Step 2: Running LOSO Cross-Validation..."
    python3 loso_cv.py

    echo "----------------------------------------"
    echo "LOSO-CV pipeline finished at $(date)"
} 2>&1 | tee loso_cv_log.txt

kill $CAFFEINE_PID 2>/dev/null
echo "Done. Caffeinate stopped."
