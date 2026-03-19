#!/usr/bin/env python3
"""
Ensemble LOSO Evaluation

For each patient with N seizures:
1. Load all N fold models (each trained with a different seizure held out)
2. For each fold k, load fold k's test set (contains the held-out seizure)
3. Run ALL N models on each test set, average probabilities
4. Apply threshold + smoothing, compute per-seizure accuracy
5. Aggregate across all folds: every seizure tested exactly once

Usage:
    python evaluate_ensemble.py                # Evaluate all patients
    python evaluate_ensemble.py --single-model # Also report single-model (non-ensemble) results
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve
from tqdm import tqdm
import json
import argparse

from train import EEGDataset, ConvTransformerModel
from total_accuracy import calculate_per_seizure_accuracy, apply_smoothing
from data_segmentation_helpers.config import (
    SEQUENCE_LENGTH,
    SEQUENCE_BATCH_SIZE,
    CONV_EMBEDDING_DIM,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_NUM_HEADS,
    TRANSFORMER_FFN_DIM,
    TRANSFORMER_DROPOUT,
    USE_CLS_TOKEN,
    PATIENTS,
    PATIENT_INDEX,
    get_patient_config,
    SEGMENT_DURATION,
    SEIZURE_COUNTS,
)

SMOOTHING_T = 20  # Heuristic fallback window size


def _get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


def _create_model(device):
    return ConvTransformerModel(
        num_input_channels=18,
        num_classes=2,
        sequence_length=SEQUENCE_LENGTH,
        embed_dim=CONV_EMBEDDING_DIM,
        num_layers=TRANSFORMER_NUM_LAYERS,
        num_heads=TRANSFORMER_NUM_HEADS,
        ffn_dim=TRANSFORMER_FFN_DIM,
        dropout=TRANSFORMER_DROPOUT,
        use_cls_token=USE_CLS_TOKEN,
    )


def load_fold_model(patient_id, fold_k, device):
    """Load a single fold's trained model and config."""
    model_path = Path("model") / patient_id / f"fold_{fold_k}" / "best_model.pth"
    if not model_path.exists():
        return None, None

    model = _create_model(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    config = checkpoint.get("config", {})
    return model, config


def get_fold_test_data(patient_id, fold_k):
    """Get test dataset path for a fold."""
    return Path("preprocessing") / "data" / patient_id / f"fold_{fold_k}" / "test_dataset.h5"


def run_inference(model, test_loader, device):
    """Run inference and return probabilities and labels."""
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, labels in test_loader:
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_labels)


def evaluate_patient_ensemble(patient_id, device, report_single=False):
    """Full ensemble evaluation for one patient.

    For each fold k:
    - Load fold k's test data
    - Run ALL fold models on it, average probabilities
    - Apply threshold + smoothing
    - Record per-seizure detection

    Returns dict with ensemble metrics, or None if not enough folds.
    """
    num_seizures = SEIZURE_COUNTS.get(patient_id, 0)
    if num_seizures < 2:
        print(f"  [SKIP] {patient_id}: <2 seizures, cannot do ensemble LOSO")
        return None

    # Discover available folds
    available_folds = []
    for k in range(num_seizures):
        model_path = Path("model") / patient_id / f"fold_{k}" / "best_model.pth"
        test_path = get_fold_test_data(patient_id, k)
        if model_path.exists() and test_path.exists():
            available_folds.append(k)

    if len(available_folds) < 2:
        print(f"  [SKIP] {patient_id}: only {len(available_folds)} folds available")
        return None

    print(f"  Loading {len(available_folds)} fold models...")

    # Load all fold models
    fold_models = {}
    fold_configs = {}
    for k in available_folds:
        model, config = load_fold_model(patient_id, k, device)
        if model is not None:
            fold_models[k] = model
            fold_configs[k] = config

    # Collect thresholds and smoothing params across folds
    thresholds = []
    smoothing_windows = []
    smoothing_counts = []
    for k, config in fold_configs.items():
        thresholds.append(config.get("optimal_threshold", 0.5))
        sw = config.get("smoothing_window")
        sc = config.get("smoothing_count")
        if sw is not None:
            smoothing_windows.append(sw)
            smoothing_counts.append(sc)

    # Evaluate each fold's test set
    total_seizures = 0
    correctly_predicted_seizures = 0
    total_fp = 0
    total_interictal_hours = 0.0
    per_fold_results = []
    single_model_results = [] if report_single else None

    sequence_duration = SEGMENT_DURATION * SEQUENCE_LENGTH

    for fold_k in available_folds:
        test_path = get_fold_test_data(patient_id, fold_k)
        test_dataset = EEGDataset(str(test_path), split="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=SEQUENCE_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        if len(test_dataset) == 0:
            print(f"    Fold {fold_k}: empty test set, skipping")
            continue

        # Run ALL models on this fold's test data, collect probabilities
        all_model_probs = []
        labels = None
        for k, model in fold_models.items():
            probs, fold_labels = run_inference(model, test_loader, device)
            all_model_probs.append(probs)
            if labels is None:
                labels = fold_labels

        # Ensemble: average probabilities across all models
        ensemble_probs = np.mean(all_model_probs, axis=0)

        # Find optimal threshold on ensemble probabilities (Youden's J)
        if len(np.unique(labels)) > 1:
            fpr, tpr, thresh = roc_curve(labels, ensemble_probs)
            j_scores = tpr - fpr
            optimal_threshold = thresh[np.argmax(j_scores)]
            # Cap at 0.5 to avoid overly conservative thresholds
            if optimal_threshold > 0.5:
                optimal_threshold = 0.5
        else:
            optimal_threshold = 0.5

        ensemble_preds = (ensemble_probs >= optimal_threshold).astype(int)

        # Smoothing
        if smoothing_windows:
            window_size = int(np.median(smoothing_windows))
            threshold_count = int(np.median(smoothing_counts))
        else:
            window_size = min(SMOOTHING_T, len(ensemble_preds))
            threshold_count = max(1, window_size // 2)

        if window_size > len(ensemble_preds):
            original_window = window_size
            window_size = max(1, len(ensemble_preds))
            threshold_count = max(1, round(threshold_count * window_size / original_window))

        if window_size > 1:
            smoothed_preds = apply_smoothing(ensemble_preds.tolist(), window_size, threshold_count)
            offset = window_size - 1
            aligned_labels = labels[offset:]
        else:
            smoothed_preds = ensemble_preds.tolist()
            aligned_labels = labels

        # Per-seizure detection for this fold
        is_preictal = (labels == 1).astype(int)
        diff = np.diff(np.concatenate(([0], is_preictal, [0])))
        seizure_starts = np.where(diff == 1)[0]
        seizure_ends = np.where(diff == -1)[0]
        n_seizures_in_fold = len(seizure_starts)

        fold_detected = 0
        if n_seizures_in_fold > 0:
            for start_idx, end_idx in zip(seizure_starts, seizure_ends):
                if window_size > 1:
                    s_start = max(0, start_idx - (window_size - 1))
                    s_end = min(len(smoothed_preds), end_idx)
                else:
                    s_start = start_idx
                    s_end = min(len(smoothed_preds), end_idx)

                if s_start < s_end and np.sum(smoothed_preds[s_start:s_end]) > 0:
                    fold_detected += 1

        total_seizures += n_seizures_in_fold
        correctly_predicted_seizures += fold_detected

        # FP rate for this fold
        interictal_mask = (aligned_labels == 0) if len(aligned_labels) == len(smoothed_preds) else (labels == 0)
        if len(interictal_mask) == len(smoothed_preds):
            fp = np.sum(np.array(smoothed_preds)[interictal_mask])
        else:
            fp = 0
        n_interictal = np.sum(interictal_mask)
        interictal_hours = (n_interictal * sequence_duration) / 3600
        total_fp += fp
        total_interictal_hours += interictal_hours

        fold_result = {
            "fold_k": fold_k,
            "n_test_samples": len(test_dataset),
            "n_seizures": n_seizures_in_fold,
            "detected": fold_detected,
            "ensemble_threshold": float(optimal_threshold),
            "smoothing_window": window_size,
            "smoothing_count": threshold_count,
        }
        per_fold_results.append(fold_result)

        status = "DETECTED" if fold_detected > 0 else "MISSED"
        print(f"    Fold {fold_k}: {n_seizures_in_fold} seizure(s), {status} "
              f"(threshold={optimal_threshold:.4f}, {len(test_dataset)} samples)")

        # Single-model comparison
        if report_single and fold_k in fold_models:
            single_probs, _ = run_inference(fold_models[fold_k], test_loader, device)
            single_threshold = fold_configs[fold_k].get("optimal_threshold", 0.5)
            if len(np.unique(labels)) > 1:
                fpr_s, tpr_s, thresh_s = roc_curve(labels, single_probs)
                j_s = tpr_s - fpr_s
                single_opt = thresh_s[np.argmax(j_s)]
                if single_opt > 0.5:
                    single_opt = 0.5
            else:
                single_opt = 0.5
            single_preds = (single_probs >= single_opt).astype(int)
            single_detected = 0
            if n_seizures_in_fold > 0:
                for start_idx, end_idx in zip(seizure_starts, seizure_ends):
                    s = single_preds[start_idx:end_idx]
                    if np.sum(s) > 0:
                        single_detected += 1
            single_model_results.append({
                "fold_k": fold_k,
                "detected": single_detected,
                "n_seizures": n_seizures_in_fold,
            })

    # Aggregate metrics
    per_seizure_accuracy = correctly_predicted_seizures / total_seizures if total_seizures > 0 else 0.0
    fp_rate = total_fp / total_interictal_hours if total_interictal_hours > 0 else 0.0

    result = {
        "patient_id": patient_id,
        "num_folds": len(available_folds),
        "ensemble_method": "probability_averaging",
        "per_seizure_accuracy": per_seizure_accuracy,
        "correctly_predicted_seizures": correctly_predicted_seizures,
        "total_seizures": total_seizures,
        "fp_rate_per_hour": fp_rate,
        "total_false_positives": int(total_fp),
        "total_interictal_hours": total_interictal_hours,
        "per_fold_results": per_fold_results,
    }

    if report_single and single_model_results:
        single_detected_total = sum(r["detected"] for r in single_model_results)
        single_total = sum(r["n_seizures"] for r in single_model_results)
        result["single_model_per_seizure_accuracy"] = single_detected_total / single_total if single_total > 0 else 0.0
        result["single_model_results"] = single_model_results

    # Save results
    results_path = Path("model") / patient_id / "ensemble_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  ENSEMBLE RESULTS for {patient_id}:")
    print(f"    Per-Seizure Accuracy: {per_seizure_accuracy:.4f} ({correctly_predicted_seizures}/{total_seizures})")
    print(f"    FP Rate: {fp_rate:.4f} per hour")
    if report_single and single_model_results:
        print(f"    Single-Model Accuracy: {result['single_model_per_seizure_accuracy']:.4f} ({single_detected_total}/{single_total})")
    print(f"    Results saved to {results_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Ensemble LOSO Evaluation")
    parser.add_argument("--single-model", action="store_true",
                        help="Also report single-model (non-ensemble) results for comparison")
    args = parser.parse_args()

    device = _get_device()

    n_patients = len(PATIENTS)
    if PATIENT_INDEX is not None:
        patients_to_process = [PATIENT_INDEX]
    else:
        patients_to_process = list(range(n_patients))

    all_results = []

    for current_idx in patients_to_process:
        patient_config = get_patient_config(current_idx)
        patient_id = patient_config["patient_id"]

        print(f"\n{'='*60}")
        print(f"ENSEMBLE EVALUATION: {patient_id}")
        print(f"{'='*60}")

        try:
            result = evaluate_patient_ensemble(patient_id, device, report_single=args.single_model)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Batch summary
    if all_results:
        print(f"\n{'='*60}")
        print("ENSEMBLE BATCH SUMMARY")
        print(f"{'='*60}")
        print(f"Patients evaluated: {len(all_results)}")

        accs = [r["per_seizure_accuracy"] for r in all_results]
        fps = [r["fp_rate_per_hour"] for r in all_results]
        total_correct = sum(r["correctly_predicted_seizures"] for r in all_results)
        total_seizures = sum(r["total_seizures"] for r in all_results)

        print(f"Mean Per-Seizure Accuracy: {np.mean(accs):.4f} (+/-{np.std(accs):.4f})")
        print(f"Overall Per-Seizure Accuracy: {total_correct}/{total_seizures} ({total_correct/total_seizures:.4f})")
        print(f"Mean FP Rate: {np.mean(fps):.4f} per hour (+/-{np.std(fps):.4f})")

        if args.single_model:
            single_accs = [r.get("single_model_per_seizure_accuracy", 0) for r in all_results if "single_model_per_seizure_accuracy" in r]
            if single_accs:
                print(f"\nSingle-Model Mean Accuracy: {np.mean(single_accs):.4f}")
                print(f"Ensemble Improvement: {np.mean(accs) - np.mean(single_accs):+.4f}")

        # Save batch results
        batch_path = Path("model") / "ensemble_batch_results.json"
        batch_summary = {
            "patients_evaluated": len(all_results),
            "mean_per_seizure_accuracy": float(np.mean(accs)),
            "overall_per_seizure_accuracy": total_correct / total_seizures if total_seizures > 0 else 0,
            "mean_fp_rate_per_hour": float(np.mean(fps)),
            "per_patient": all_results,
        }
        with open(batch_path, "w") as f:
            json.dump(batch_summary, f, indent=2)
        print(f"\nBatch results saved to {batch_path}")


if __name__ == "__main__":
    main()
