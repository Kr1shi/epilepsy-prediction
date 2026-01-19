#!/bin/bash

# ==========================================
# EEG Seizure Prediction Pipeline Runner
# ==========================================

# Stop execution immediately if any command fails
set -e

echo "=========================================="
echo "🚀 STARTING FULL PIPELINE EXECUTION"
echo "=========================================="
echo "Date: $(date)"
echo "=========================================="

# 1. Data Segmentation
echo ""
echo "------------------------------------------"
echo "STEP 1: SEGMENTATION (data_segmentation.py)"
echo "------------------------------------------"
python3 data_segmentation.py

# 2. Data Preprocessing
echo ""
echo "------------------------------------------"
echo "STEP 2: PREPROCESSING (data_preprocessing.py)"
echo "------------------------------------------"
python3 data_preprocessing.py

# 3. Model Training
echo ""
echo "------------------------------------------"
echo "STEP 3: TRAINING (train.py)"
echo "------------------------------------------"
python3 train.py

# 4. Evaluation
echo ""
echo "------------------------------------------"
echo "STEP 4: EVALUATION (evaluate_test.py)"
echo "------------------------------------------"
python3 evaluate_test.py

echo ""
echo "=========================================="
echo "✅ PIPELINE COMPLETED SUCCESSFULLY"
echo "=========================================="
