#!/usr/bin/env python3
"""
Evaluate trained model on test dataset with Temporal Sorting and Smoothing.

Features:
1. Sorts test data chronologically (File -> Start Time).
2. Applies "M-of-N" smoothing (N positives in last T frames).
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import h5py
from pathlib import Path
from collections import deque
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from tqdm import tqdm
import json
import argparse

# Import from train.py
from train import EEGDataset, CNN_LSTM_Hybrid, MetricsTracker
from data_segmentation_helpers.config import *

# ==========================================
# CONFIGURATION
# ==========================================
# T: Size of the rolling window
SMOOTHING_WINDOW_SIZE = 5  

# N: Number of positives required in the window to trigger a detection
SMOOTHING_REQUIRED_POSITIVES = 3  

# Example: 3/5 means "Trigger alarm if 3 out of the last 5 segments were positive"
# ==========================================

def get_positive_label():
    """Get positive class label based on task mode"""
    return 'preictal' if TASK_MODE == 'prediction' else 'ictal'

def get_sorted_indices(h5_path):
    """
    Reads metadata from HDF5 and returns indices that sort the data
    chronologically (grouped by file, then by start time).
    """
    print(f"Reading metadata for sorting from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        # Check if segment_info exists
        if 'segment_info' not in f:
            print("⚠️ Warning: 'segment_info' not found in HDF5. Cannot sort by time. Using default order.")
            return list(range(len(f['spectrograms'])))
            
        # Read metadata
        start_times = f['segment_info']['start_times'][:]
        file_names = [n.decode('utf-8') for n in f['segment_info']['file_names'][:]]
        
        # Create a list of tuples: (file_name, start_time, original_index)
        meta_list = []
        for i in range(len(start_times)):
            meta_list.append({
                'index': i,
                'file': file_names[i],
                'time': start_times[i]
            })
            
    # Sort: Primary key = File Name, Secondary key = Start Time
    print("Sorting test data chronologically...")
    sorted_meta = sorted(meta_list, key=lambda x: (x['file'], x['time']))
    
    # Extract sorted indices
    sorted_indices = [item['index'] for item in sorted_meta]
    return sorted_indices

def apply_smoothing(predictions, window_size, threshold):
    """
    Applies M-of-N smoothing to a sequence of predictions.
    
    Args:
        predictions (list/array): Raw binary predictions (0 or 1)
        window_size (int): T - size of looking back window
        threshold (int): N - required positives
    
    Returns:
        np.array: Smoothed binary predictions
    """
    smoothed_preds = []
    # Queue to hold the history of the last T predictions
    window = deque(maxlen=window_size)
    
    for p in predictions:
        window.append(p)
        # If the buffer isn't full yet, we can either be strict or loose.
        # Here we check: do we have enough positives in the *current* buffer?
        if sum(window) >= threshold:
            smoothed_preds.append(1)
        else:
            smoothed_preds.append(0)
            
    return np.array(smoothed_preds)

def evaluate_model(model_path, test_data_path, device):
    """
    Load model, sort data, evaluate, and smooth.
    """
    # 1. Load Dataset
    print(f"Loading test dataset from {test_data_path}...")
    full_dataset = EEGDataset(test_data_path, split='test')
    
    # 2. Get Sorted Indices
    sorted_indices = get_sorted_indices(test_data_path)
    
    # 3. Create Sorted Subset
    sorted_dataset = Subset(full_dataset, sorted_indices)
    
    test_loader = DataLoader(
        sorted_dataset,
        batch_size=SEQUENCE_BATCH_SIZE,
        shuffle=False, # STRICTLY FALSE to preserve sorted order
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 4. Initialize Model
    print(f"Initializing Deep CNN-BiLSTM model...")
    model = CNN_LSTM_Hybrid(
        num_input_channels=18,
        num_classes=2,
        sequence_length=SEQUENCE_LENGTH,
        cnn_feature_dim=512,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        lstm_num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT
    )

    # 5. Load Weights
    print(f"Loading model checkpoint from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Get config
    checkpoint_task_mode = checkpoint.get('config', {}).get('task_mode', TASK_MODE)
    positive_class = checkpoint.get('config', {}).get('positive_class', get_positive_label())

    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Smoothing Config: Require {SMOOTHING_REQUIRED_POSITIVES} positives in last {SMOOTHING_WINDOW_SIZE} frames ({SMOOTHING_REQUIRED_POSITIVES}/{SMOOTHING_WINDOW_SIZE})")

    # 6. Evaluate Loop
    print("\nRunning inference...")
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    num_batches = 0

    all_raw_predictions = []
    all_true_labels = []
    all_probabilities = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for spectrograms, labels in pbar:
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            # Forward pass
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            num_batches += 1

            # Get Raw Predictions
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            # Store (CPU)
            all_raw_predictions.extend(preds.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    # 7. Apply Smoothing
    print("\nApplying temporal smoothing...")
    all_smoothed_predictions = apply_smoothing(
        all_raw_predictions, 
        SMOOTHING_WINDOW_SIZE, 
        SMOOTHING_REQUIRED_POSITIVES
    )

    # 8. Compute Metrics (Raw vs Smoothed)
    avg_loss = total_loss / num_batches
    
    # Helper to compute dict of metrics
    def compute_stats(y_true, y_pred, y_prob=None):
        m = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        if y_prob is not None:
            try:
                m['auc_roc'] = roc_auc_score(y_true, y_prob)
            except:
                m['auc_roc'] = 0.0
        else:
            m['auc_roc'] = 0.0 # AUC not applicable for smoothed binary preds usually
        return m

    raw_metrics = compute_stats(all_true_labels, all_raw_predictions, all_probabilities)
    raw_metrics['loss'] = avg_loss
    
    smoothed_metrics = compute_stats(all_true_labels, all_smoothed_predictions)
    smoothed_metrics['loss'] = avg_loss # Same loss

    # Confusion Matrices
    cm_raw = confusion_matrix(all_true_labels, all_raw_predictions)
    cm_smoothed = confusion_matrix(all_true_labels, all_smoothed_predictions)

    return (
        raw_metrics, smoothed_metrics, 
        cm_raw, cm_smoothed, 
        all_true_labels, all_raw_predictions, all_smoothed_predictions,
        checkpoint_task_mode, positive_class
    )

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on test dataset')
    parser.add_argument('--epoch', type=int, default=TRAINING_EPOCHS, help='Epoch number to evaluate')
    args = parser.parse_args()

    # Determine folds
    if LOOCV_FOLD_ID is None:
        folds_to_process = list(range(LOOCV_TOTAL_SEIZURES))
        print("BATCH EVALUATION: ALL FOLDS")
    else:
        folds_to_process = [LOOCV_FOLD_ID]
        print(f"SINGLE FOLD EVALUATION: Fold {LOOCV_FOLD_ID}")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    batch_results = {}

    for current_fold in folds_to_process:
        fold_config = get_fold_config(current_fold)
        current_output_prefix = fold_config['output_prefix']

        print(f"\n{'='*60}")
        print(f"EVALUATING FOLD {current_fold}")
        print(f"{'='*60}")

        try:
            model_path = Path(f"model/{current_output_prefix}/epoch_{args.epoch:03d}.pth")
            test_data_path = Path("preprocessing") / "data" / current_output_prefix / "test_dataset.h5"

            if not model_path.exists() or not test_data_path.exists():
                print(f"❌ Files missing for fold {current_fold}. Skipping.")
                continue

            # Run Evaluation
            (raw, smoothed, cm_raw, cm_smoothed, 
             y_true, y_raw, y_smooth, 
             mode, pos_class) = evaluate_model(model_path, test_data_path, device)

            # Print Comparison
            print("\n" + "="*60)
            print("RESULTS COMPARISON")
            print("="*60)
            print(f"{'METRIC':<12} | {'RAW':<10} | {'SMOOTHED (M-of-N)':<10}")
            print("-" * 40)
            print(f"{'Accuracy':<12} | {raw['accuracy']:.4f}     | {smoothed['accuracy']:.4f}")
            print(f"{'Precision':<12} | {raw['precision']:.4f}     | {smoothed['precision']:.4f}")
            print(f"{'Recall':<12} | {raw['recall']:.4f}     | {smoothed['recall']:.4f}")
            print(f"{'F1 Score':<12} | {raw['f1']:.4f}     | {smoothed['f1']:.4f}")
            print("-" * 40)
            
            print(f"\nConfiguration: {SMOOTHING_REQUIRED_POSITIVES} positives required in last {SMOOTHING_WINDOW_SIZE} frames.")
            
            print("\nSmoothed Confusion Matrix:")
            print(f"                Predicted")
            print(f"                Interictal  {pos_class.capitalize()}")
            print(f"Actual Interictal    {cm_smoothed[0,0]:6d}    {cm_smoothed[0,1]:6d}")
            print(f"       {pos_class.capitalize():9s}   {cm_smoothed[1,0]:6d}    {cm_smoothed[1,1]:6d}")

            # Save Results
            fold_results = {
                'fold': current_fold,
                'config': {'window': SMOOTHING_WINDOW_SIZE, 'threshold': SMOOTHING_REQUIRED_POSITIVES},
                'raw_metrics': raw,
                'smoothed_metrics': smoothed,
                'cm_smoothed': cm_smoothed.tolist()
            }
            
            out_path = Path(f"model/{current_output_prefix}/test_results_smoothed.json")
            with open(out_path, 'w') as f:
                json.dump(fold_results, f, indent=2)
            
            batch_results[current_fold] = fold_results

        except Exception as e:
            print(f"❌ Error in fold {current_fold}: {e}")
            import traceback
            traceback.print_exc()

    # Batch Summary
    if len(batch_results) > 0 and LOOCV_FOLD_ID is None:
        print("\n" + "="*60)
        print("BATCH SUMMARY (SMOOTHED METRICS)")
        print("="*60)
        s_f1 = [r['smoothed_metrics']['f1'] for r in batch_results.values()]
        s_acc = [r['smoothed_metrics']['accuracy'] for r in batch_results.values()]
        
        print(f"Mean Smoothed F1:       {np.mean(s_f1):.4f} (±{np.std(s_f1):.4f})")
        print(f"Mean Smoothed Accuracy: {np.mean(s_acc):.4f} (±{np.std(s_acc):.4f})")
        
        with open("model/batch_test_results_smoothed.json", 'w') as f:
            json.dump(batch_results, f, indent=2)

if __name__ == "__main__":
    main()