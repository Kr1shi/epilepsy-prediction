"""Visualize model predictions on the patient timeline.

For each patient, loads every per-seizure model and runs inference on the
corresponding test fold. Plots predicted probabilities along the timeline,
overlaid with ground truth seizures, preictal zones, and decision thresholds.

Usage:
    python visualize_predictions.py              # All patients
    python visualize_predictions.py --patient 0  # Single patient (by index)
"""

import argparse
import json
import glob
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm

from train import EEGDataset, ConvGRUModel, create_datasets, get_datasets_per_patient, apply_normalization_to_datasets
from data_segmentation_helpers.config import *
from visualize_segmentation import load_patient_timeline, build_file_offsets, _parse_time_str
from data_segmentation_helpers.segmentation import parse_summary_file

OUTPUT_DIR = Path("preprocessing") / "visualizations" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_model(device):
    model = ConvGRUModel(
        num_input_channels=18,
        num_classes=2,
        embed_dim=CONV_EMBEDDING_DIM,
        gru_hidden=GRU_HIDDEN_DIM,
        gru_layers=GRU_NUM_LAYERS,
        dropout=GRU_DROPOUT,
    )
    return model


def get_predictions(model_path, test_loader, device):
    """Run inference with a specific model checkpoint. Returns probs, labels, threshold."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    threshold = config.get("optimal_threshold", 0.5)

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

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.array(all_probs), np.array(all_labels), threshold


def get_sequence_global_times(patient_id, sequences):
    """Convert sequence local times to global timeline positions."""
    file_offsets = build_file_offsets(patient_id)
    seq_duration = SEGMENT_DURATION * SEQUENCE_LENGTH

    global_centers = []
    for seq in sequences:
        start_sec = seq["sequence_start_sec"]
        file_offset = file_offsets.get(seq["file"], 0)
        global_start = file_offset + start_sec
        global_end = global_start + seq_duration
        global_centers.append((global_start + global_end) / 2)

    return np.array(global_centers)


def plot_patient_predictions(patient_id, seizures, total_duration, fold_results):
    """Plot timeline with model prediction probabilities for each seizure fold.

    Args:
        fold_results: list of dicts with keys:
            seizure_id, probs, labels, threshold, global_times
    """
    to_hr = lambda s: s / 3600

    fig, axes = plt.subplots(len(fold_results) + 1, 1,
                              figsize=(20, 3 * (len(fold_results) + 1)),
                              sharex=True)
    if len(fold_results) + 1 == 1:
        axes = [axes]

    # Top panel: ground truth timeline
    ax = axes[0]
    ax.set_title(f"{patient_id} — Ground Truth Timeline", fontsize=12, fontweight="bold")

    # Draw preictal zones
    for sz in seizures:
        pre_start = max(0, sz["start"] - PREICTAL_WINDOW)
        pre_end = sz["start"] - PREICTAL_ONSET_BUFFER
        if pre_end > pre_start:
            ax.axvspan(to_hr(pre_start), to_hr(pre_end), color="#ffcdd2", alpha=0.5)

    # Draw seizures
    for sz in seizures:
        ax.axvspan(to_hr(sz["start"]), to_hr(sz["end"]), color="#d32f2f", alpha=0.9)
        ax.text(to_hr((sz["start"] + sz["end"]) / 2), 0.9, f'sz{sz["id"]}',
                ha="center", va="top", fontsize=8, color="#d32f2f", fontweight="bold")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Ground Truth")
    ax.set_yticks([])

    handles = [
        mpatches.Patch(color="#d32f2f", alpha=0.9, label="Seizure"),
        mpatches.Patch(color="#ffcdd2", alpha=0.5, label="Preictal zone"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7)

    # Per-fold prediction panels
    for i, fold in enumerate(fold_results):
        ax = axes[i + 1]
        sid = fold["seizure_id"]
        probs = fold["probs"]
        labels = fold["labels"]
        threshold = fold["threshold"]
        times_hr = np.array([to_hr(t) for t in fold["global_times"]])

        # Color points by ground truth label
        preictal_mask = labels == 1
        interictal_mask = labels == 0

        # Plot probabilities as scatter
        if np.any(interictal_mask):
            ax.scatter(times_hr[interictal_mask], probs[interictal_mask],
                       c="#4caf50", s=6, alpha=0.5, zorder=3, label="Interictal")
        if np.any(preictal_mask):
            ax.scatter(times_hr[preictal_mask], probs[preictal_mask],
                       c="#e53935", s=6, alpha=0.5, zorder=3, label="Preictal")

        # Threshold line
        ax.axhline(y=threshold, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(to_hr(total_duration) * 0.99, threshold + 0.03,
                f"thresh={threshold:.2f}", ha="right", fontsize=7, color="black")

        # Draw seizure locations for reference
        for sz in seizures:
            ax.axvspan(to_hr(sz["start"]), to_hr(sz["end"]), color="#d32f2f", alpha=0.15)

        # Metrics annotation
        auc = fold.get("auc", 0)
        acc = fold.get("accuracy", 0)
        ax.set_title(f"Test Seizure {sid}  |  AUC={auc:.3f}  Acc={acc:.3f}",
                     fontsize=10, loc="left")

        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("P(preictal)")
        ax.grid(axis="y", alpha=0.2)

        if i == 0:
            ax.legend(loc="upper right", fontsize=7, markerscale=2)

    axes[-1].set_xlabel("Time (hours)")
    axes[-1].set_xlim(0, to_hr(total_duration))

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{patient_id}_predictions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions on timeline")
    parser.add_argument("--patient", type=int, default=None, help="Patient index")
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

    for current_idx in patients_to_process:
        patient_id = PATIENTS[current_idx]
        patient_config = get_patient_config(current_idx)
        dataset_prefix = patient_config["output_prefix"]

        print(f"\nProcessing {patient_id}...")

        # Load timeline
        try:
            seizures, total_duration = load_patient_timeline(patient_id)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        # Load segmentation JSON for sequence metadata (global times)
        json_path = f"{dataset_prefix}_sequences_prediction.json"
        if not Path(json_path).exists():
            print(f"  Skipping: no segmentation JSON")
            continue

        with open(json_path) as f:
            all_seqs = json.load(f)["sequences"]

        # Load datasets with same filtering as training
        try:
            all_datasets = create_datasets([current_idx], skip_missing_class=True)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        patient_datasets = all_datasets.get(patient_id, {})
        if len(patient_datasets) < 2:
            print(f"  Skipping: need at least 2 valid folds, got {len(patient_datasets)}")
            continue

        # Group sequences by split (seizure ID)
        seqs_by_split = {}
        for seq in all_seqs:
            sp = seq.get("split")
            if sp is not None:
                seqs_by_split.setdefault(sp, []).append(seq)

        fold_results = []

        for sid in sorted(patient_datasets.keys(), key=str):
            model_dir = Path("model") / "per_patient" / dataset_prefix / f"test_seizure_{sid}"
            model_path = model_dir / "best_model.pth"

            if not model_path.exists():
                continue

            # Compute normalization from training folds only (exclude test fold)
            train_ds_list = [patient_datasets[s] for s in patient_datasets if s != sid]
            test_ds_list = [patient_datasets[sid]]
            if not train_ds_list:
                continue
            apply_normalization_to_datasets(train_ds_list, test_ds_list)

            # Test data for this fold
            test_ds = patient_datasets[sid]
            test_loader = DataLoader(test_ds, batch_size=SEQUENCE_BATCH_SIZE, shuffle=False, num_workers=0)

            probs, labels, threshold = get_predictions(model_path, test_loader, device)

            # Get global times for test sequences
            # The test fold sequences are those with split == sid
            fold_seqs = seqs_by_split.get(sid, seqs_by_split.get(int(sid) if sid.isdigit() else sid, []))
            global_times = get_sequence_global_times(patient_id, fold_seqs)

            # Truncate to match (in case balancing dropped some)
            min_len = min(len(probs), len(global_times))
            probs = probs[:min_len]
            labels = labels[:min_len]
            global_times = global_times[:min_len]

            # Compute metrics
            if len(np.unique(labels)) > 1:
                from sklearn.metrics import roc_auc_score, accuracy_score
                auc = roc_auc_score(labels, probs)
                preds = (probs >= threshold).astype(int)
                acc = accuracy_score(labels, preds)
            else:
                auc = 0.5
                acc = 0.5

            fold_results.append({
                "seizure_id": sid,
                "probs": probs,
                "labels": labels,
                "threshold": threshold,
                "global_times": global_times,
                "auc": auc,
                "accuracy": acc,
            })

        if fold_results:
            plot_patient_predictions(patient_id, seizures, total_duration, fold_results)
        else:
            print(f"  No trained models found")

        gc.collect()


if __name__ == "__main__":
    main()
