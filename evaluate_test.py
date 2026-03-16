#!/usr/bin/env python3
"""
Evaluate trained models on test seizure folds with proper thresholding.

For each patient, loads best_model.pth for each seizure fold,
applies the optimal threshold from training, and reports metrics.

Usage:
    python evaluate_test.py              # Evaluate all patients
    python evaluate_test.py --patient 0  # Evaluate single patient (by index)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from tqdm import tqdm
import json
import argparse
import os
import gc

from train import EEGDataset, ConvGRUModel, MetricsTracker, create_datasets, get_datasets_per_patient
from data_segmentation_helpers.config import *


def create_model(device):
    """Create a ConvGRUModel instance."""
    return ConvGRUModel(
        num_input_channels=18,
        num_classes=2,
        embed_dim=CONV_EMBEDDING_DIM,
        gru_hidden=GRU_HIDDEN_DIM,
        gru_layers=GRU_NUM_LAYERS,
        dropout=GRU_DROPOUT,
    )


def evaluate_fold(model_path, test_loader, device):
    """Load model checkpoint and evaluate on a test fold.

    Returns dict with all metrics, using the optimal threshold from training.
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    optimal_threshold = config.get("optimal_threshold", 0.5)
    smoothing_window = config.get("smoothing_window", 1)
    smoothing_count = config.get("smoothing_count", 1)

    # Create and load model
    model = create_model(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, labels in test_loader:
            x, labels = x.to(device), labels.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    probs_np = np.array(all_probs)
    labels_np = np.array(all_labels)

    # Metrics at default threshold (0.5)
    preds_default = (probs_np >= 0.5).astype(int)

    # Metrics at optimal threshold
    preds_optimal = (probs_np >= optimal_threshold).astype(int)

    # AUC (threshold-independent)
    if len(np.unique(labels_np)) > 1:
        auc = roc_auc_score(labels_np, probs_np)
    else:
        auc = 0.5

    # Apply smoothing to optimal-threshold predictions
    smoothed_preds = preds_optimal
    smoothed_labels = labels_np
    if smoothing_window > 1 and len(preds_optimal) >= smoothing_window:
        kernel = np.ones(smoothing_window)
        smoothed_sums = np.convolve(preds_optimal, kernel, mode="valid")
        smoothed_preds = (smoothed_sums >= smoothing_count).astype(int)
        smoothed_labels = labels_np[smoothing_window - 1:]

    return {
        "n_samples": len(labels_np),
        "optimal_threshold": optimal_threshold,
        "smoothing_window": smoothing_window,
        "smoothing_count": smoothing_count,
        "auc_roc": auc,
        # Default threshold metrics
        "default_accuracy": accuracy_score(labels_np, preds_default),
        "default_f1": f1_score(labels_np, preds_default, zero_division=0),
        # Optimal threshold metrics
        "accuracy": accuracy_score(labels_np, preds_optimal),
        "precision": precision_score(labels_np, preds_optimal, zero_division=0),
        "recall": recall_score(labels_np, preds_optimal, zero_division=0),
        "f1": f1_score(labels_np, preds_optimal, zero_division=0),
        # Smoothed metrics
        "smoothed_accuracy": accuracy_score(smoothed_labels, smoothed_preds),
        "smoothed_f1": f1_score(smoothed_labels, smoothed_preds, zero_division=0),
        "smoothed_precision": precision_score(smoothed_labels, smoothed_preds, zero_division=0),
        "smoothed_recall": recall_score(smoothed_labels, smoothed_preds, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Conv-GRU models")
    parser.add_argument("--patient", type=int, default=None, help="Patient index to evaluate (default: all)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if args.patient is not None:
        patients_to_process = [args.patient]
    else:
        patients_to_process = list(range(len(PATIENTS)))

    print("=" * 100)
    print("EVALUATION WITH OPTIMAL THRESHOLDING")
    print("=" * 100)

    all_results = []

    for current_idx in patients_to_process:
        patient_id = PATIENTS[current_idx]
        patient_config = get_patient_config(current_idx)
        dataset_prefix = patient_config["output_prefix"]

        # Load all datasets for this patient
        try:
            all_datasets = create_datasets([current_idx], skip_missing_class=False)
        except Exception as e:
            print(f"  Skipping {patient_id}: {e}")
            continue

        patient_datasets = all_datasets.get(patient_id, {})
        if not patient_datasets:
            print(f"  Skipping {patient_id}: no datasets")
            continue

        for sid in patient_datasets:
            model_dir = Path("model") / "per_patient" / dataset_prefix / f"test_seizure_{sid}"
            model_path = model_dir / "best_model.pth"

            if not model_path.exists():
                continue

            # Build test loader for this fold
            test_ds = patient_datasets[sid]
            test_loader = DataLoader(
                test_ds, batch_size=SEQUENCE_BATCH_SIZE, shuffle=False, num_workers=0
            )

            result = evaluate_fold(model_path, test_loader, device)
            result["patient_id"] = patient_id
            result["seizure_id"] = sid
            all_results.append(result)

        gc.collect()

    if not all_results:
        print("No results found.")
        return

    # Print results table
    print(f"\n{'Patient':<10} {'Seizure':<10} {'AUC':<8} {'Acc':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'SmAcc':<8} {'SmF1':<8} {'Thresh':<8} {'N':<6}")
    print("=" * 100)

    for r in all_results:
        print(f"{r['patient_id']:<10} {r['seizure_id']:<10} "
              f"{r['auc_roc']:<8.4f} {r['accuracy']:<8.4f} {r['f1']:<8.4f} "
              f"{r['precision']:<8.4f} {r['recall']:<8.4f} "
              f"{r['smoothed_accuracy']:<8.4f} {r['smoothed_f1']:<8.4f} "
              f"{r['optimal_threshold']:<8.4f} {r['n_samples']:<6}")

    print("=" * 100)

    # Summary statistics
    aucs = [r["auc_roc"] for r in all_results]
    accs = [r["accuracy"] for r in all_results]
    f1s = [r["f1"] for r in all_results]
    sm_accs = [r["smoothed_accuracy"] for r in all_results]
    sm_f1s = [r["smoothed_f1"] for r in all_results]
    precs = [r["precision"] for r in all_results]
    recs = [r["recall"] for r in all_results]

    print(f"{'MEAN':<10} {'':10} "
          f"{np.mean(aucs):<8.4f} {np.mean(accs):<8.4f} {np.mean(f1s):<8.4f} "
          f"{np.mean(precs):<8.4f} {np.mean(recs):<8.4f} "
          f"{np.mean(sm_accs):<8.4f} {np.mean(sm_f1s):<8.4f}")
    print(f"{'STD':<10} {'':10} "
          f"{np.std(aucs):<8.4f} {np.std(accs):<8.4f} {np.std(f1s):<8.4f} "
          f"{np.std(precs):<8.4f} {np.std(recs):<8.4f} "
          f"{np.std(sm_accs):<8.4f} {np.std(sm_f1s):<8.4f}")

    print(f"\nTotal folds evaluated: {len(all_results)}")
    print(f"AUC > 0.9: {sum(1 for a in aucs if a > 0.9)}/{len(aucs)}")
    print(f"AUC > 0.8: {sum(1 for a in aucs if a > 0.8)}/{len(aucs)}")
    print(f"AUC > 0.7: {sum(1 for a in aucs if a > 0.7)}/{len(aucs)}")
    print(f"AUC < 0.5: {sum(1 for a in aucs if a < 0.5)}/{len(aucs)}")

    # Per-patient summary
    print(f"\n{'Patient':<10} {'Folds':<8} {'Mean AUC':<10} {'Mean Acc':<10} {'Mean F1':<10} {'Mean SmF1':<10}")
    print("-" * 60)
    patient_ids = sorted(set(r["patient_id"] for r in all_results))
    for pid in patient_ids:
        pr = [r for r in all_results if r["patient_id"] == pid]
        print(f"{pid:<10} {len(pr):<8} "
              f"{np.mean([r['auc_roc'] for r in pr]):<10.4f} "
              f"{np.mean([r['accuracy'] for r in pr]):<10.4f} "
              f"{np.mean([r['f1'] for r in pr]):<10.4f} "
              f"{np.mean([r['smoothed_f1'] for r in pr]):<10.4f}")

    # Save results
    output_path = Path("model") / "evaluation_results.json"
    save_data = {
        "per_fold": all_results,
        "summary": {
            "n_folds": len(all_results),
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "mean_accuracy": float(np.mean(accs)),
            "mean_f1": float(np.mean(f1s)),
            "mean_smoothed_accuracy": float(np.mean(sm_accs)),
            "mean_smoothed_f1": float(np.mean(sm_f1s)),
        },
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
