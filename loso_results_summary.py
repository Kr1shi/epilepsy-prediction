#!/usr/bin/env python3
"""
Generate CSV summary of LOSO-CV results with per-patient and overall metrics.

Usage:
    python loso_results_summary.py
    python loso_results_summary.py --results model/loso_cv/loso_cv_results.json
    python loso_results_summary.py --output results.csv
"""
import json
import csv
import argparse
import numpy as np
from pathlib import Path


def compute_patient_metrics(patient_data):
    """Compute metrics for a single patient from per-fold results."""
    seq_hours = (180 * 5) / 3600  # 0.25 hours per sequence

    fold_aucs = []
    fold_accs = []
    fold_sensitivities = []
    fold_specificities = []
    fold_fp_rates = []
    n_detected = 0
    n_folds = 0

    for fold in patient_data["per_fold"]:
        n_pos = fold["n_test_positive"]
        n_neg = fold["n_test_negative"]
        precision = fold["precision"]
        recall = fold["recall"]  # recall = sensitivity
        acc = fold["accuracy"]
        auc = fold["auc_roc"]

        # TP, FP, FN, TN from precision/recall/counts
        tp = recall * n_pos
        fn = n_pos - tp
        if precision > 0:
            fp = (tp / precision) - tp
        else:
            fp = 0
        tn = n_neg - fp

        # Specificity = TN / (TN + FP)
        if n_neg > 0:
            specificity = tn / n_neg
        else:
            specificity = 1.0

        # FP/hour
        interictal_hours = n_neg * seq_hours
        fp_rate = fp / interictal_hours if interictal_hours > 0 else 0.0

        fold_aucs.append(auc)
        fold_accs.append(acc)
        fold_sensitivities.append(recall)
        fold_specificities.append(specificity)
        fold_fp_rates.append(fp_rate)
        n_detected += fold["seizure_detected"]
        n_folds += 1

    detection_rate = n_detected / n_folds if n_folds > 0 else 0.0

    return {
        "n_folds": n_folds,
        "accuracy": np.mean(fold_accs),
        "sensitivity": np.mean(fold_sensitivities),
        "specificity": np.mean(fold_specificities),
        "auc_roc": np.mean(fold_aucs),
        "seizure_detection_rate": detection_rate,
        "fp_per_hour": np.mean(fold_fp_rates),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate CSV summary of LOSO-CV results")
    parser.add_argument(
        "--results",
        type=str,
        default="model/loso_cv/loso_cv_results.json",
        help="Path to LOSO-CV results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="loso_results_summary.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    columns = [
        "patient",
        "n_folds",
        "accuracy",
        "sensitivity",
        "specificity",
        "auc_roc",
        "seizure_detection_rate",
        "fp_per_hour",
    ]

    rows = []
    all_metrics = {k: [] for k in columns[2:]}

    for pid, pdata in data["patients"].items():
        m = compute_patient_metrics(pdata)
        row = {
            "patient": pid,
            "n_folds": m["n_folds"],
            "accuracy": f"{m['accuracy']:.4f}",
            "sensitivity": f"{m['sensitivity']:.4f}",
            "specificity": f"{m['specificity']:.4f}",
            "auc_roc": f"{m['auc_roc']:.4f}",
            "seizure_detection_rate": f"{m['seizure_detection_rate']:.4f}",
            "fp_per_hour": f"{m['fp_per_hour']:.2f}",
        }
        rows.append(row)
        for k in columns[2:]:
            all_metrics[k].append(m[k])

    # Mean row
    mean_row = {"patient": "MEAN", "n_folds": sum(r["n_folds"] for r in rows if isinstance(r["n_folds"], int))}
    # Fix: n_folds from the dict
    mean_row["n_folds"] = ""
    for k in columns[2:]:
        mean_row[k] = f"{np.mean(all_metrics[k]):.4f}"

    # Median row
    median_row = {"patient": "MEDIAN", "n_folds": ""}
    for k in columns[2:]:
        median_row[k] = f"{np.median(all_metrics[k]):.4f}"

    # Std row
    std_row = {"patient": "STD", "n_folds": ""}
    for k in columns[2:]:
        std_row[k] = f"{np.std(all_metrics[k]):.4f}"

    rows.extend([mean_row, median_row, std_row])

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # Print to console too
    header = f"{'Patient':<10} {'Folds':>5} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'AUC':>8} {'Det%':>8} {'FP/hr':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['patient']:<10} {str(row['n_folds']):>5} {row['accuracy']:>8} "
            f"{row['sensitivity']:>8} {row['specificity']:>8} {row['auc_roc']:>8} "
            f"{row['seizure_detection_rate']:>8} {row['fp_per_hour']:>8}"
        )

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
