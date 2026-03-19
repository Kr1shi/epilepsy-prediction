"""Analyze whether model prediction probability increases as we approach seizure onset.

For each seizure fold, loads the model and runs inference on the test sequences,
then plots P(preictal) vs time-to-seizure for preictal sequences only.
"""

import json
import glob
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from train import EEGDataset, ConvGRUModel, create_datasets
from data_segmentation_helpers.config import *

OUTPUT_DIR = Path("preprocessing") / "visualizations" / "temporal_trend"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_model():
    return ConvGRUModel(
        num_input_channels=18, num_classes=2,
        embed_dim=CONV_EMBEDDING_DIM, gru_hidden=GRU_HIDDEN_DIM,
        gru_layers=GRU_NUM_LAYERS, dropout=GRU_DROPOUT,
    )


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Collect all (time_to_seizure, probability) pairs across all patients/folds
    all_times = []
    all_probs = []
    per_patient = {}

    for current_idx in range(len(PATIENTS)):
        patient_id = PATIENTS[current_idx]
        patient_config = get_patient_config(current_idx)
        dataset_prefix = patient_config["output_prefix"]

        # Load segmentation JSON for time_to_seizure metadata
        json_path = f"{dataset_prefix}_sequences_prediction.json"
        if not Path(json_path).exists():
            continue

        with open(json_path) as f:
            all_seqs = json.load(f)["sequences"]

        # Group preictal sequences by split, index by position
        seqs_by_split = {}
        for seq in all_seqs:
            sp = seq.get("split")
            if sp is not None:
                seqs_by_split.setdefault(sp, []).append(seq)

        # Load datasets
        try:
            all_datasets = create_datasets([current_idx], skip_missing_class=False)
        except:
            continue

        patient_datasets = all_datasets.get(patient_id, {})
        if not patient_datasets:
            continue

        patient_times = []
        patient_probs = []

        for sid in patient_datasets:
            model_dir = Path("model") / "per_patient" / dataset_prefix / f"test_seizure_{sid}"
            model_path = model_dir / "best_model.pth"
            if not model_path.exists():
                continue

            # Load model
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model = create_model()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

            # Run inference
            test_ds = patient_datasets[sid]
            test_loader = DataLoader(test_ds, batch_size=SEQUENCE_BATCH_SIZE, shuffle=False, num_workers=0)

            probs_list = []
            labels_list = []
            with torch.no_grad():
                for x, labels in test_loader:
                    x = x.to(device)
                    outputs = model(x)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    probs_list.extend(probs.cpu().numpy())
                    labels_list.extend(labels.numpy())

            del model
            gc.collect()

            # Match predictions with time_to_seizure from JSON
            fold_seqs = seqs_by_split.get(sid, seqs_by_split.get(int(sid) if isinstance(sid, str) and sid.isdigit() else sid, []))

            min_len = min(len(probs_list), len(fold_seqs))
            for i in range(min_len):
                seq = fold_seqs[i]
                if seq["type"] == "preictal" and "time_to_seizure" in seq:
                    tts = seq["time_to_seizure"]  # seconds until seizure
                    if tts is not None and tts >= 0:
                        tts_min = tts / 60.0
                        all_times.append(tts_min)
                        all_probs.append(probs_list[i])
                        patient_times.append(tts_min)
                        patient_probs.append(probs_list[i])

        if patient_times:
            per_patient[patient_id] = (np.array(patient_times), np.array(patient_probs))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_times:
        print("No preictal predictions with time_to_seizure found.")
        return

    all_times = np.array(all_times)
    all_probs = np.array(all_probs)

    # === Plot 1: Global scatter + binned trend ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter
    ax = axes[0]
    ax.scatter(all_times, all_probs, alpha=0.15, s=8, c="#1976d2")
    ax.set_xlabel("Time to seizure onset (minutes)")
    ax.set_ylabel("P(preictal)")
    ax.set_title(f"All patients: P(preictal) vs time to seizure\n(n={len(all_times)} preictal sequences)")
    ax.set_xlim(max(all_times) + 1, -1)  # Reversed: far left = far from seizure
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.2)

    # Binned trend
    ax = axes[1]
    bin_edges = np.arange(0, max(all_times) + 5, 5)  # 5-minute bins
    bin_centers = []
    bin_means = []
    bin_stds = []
    bin_counts = []

    for i in range(len(bin_edges) - 1):
        mask = (all_times >= bin_edges[i]) & (all_times < bin_edges[i + 1])
        if np.sum(mask) >= 3:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_means.append(np.mean(all_probs[mask]))
            bin_stds.append(np.std(all_probs[mask]))
            bin_counts.append(np.sum(mask))

    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)

    ax.plot(bin_centers, bin_means, "o-", color="#d32f2f", linewidth=2, markersize=6)
    ax.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds,
                     color="#d32f2f", alpha=0.15)
    ax.set_xlabel("Time to seizure onset (minutes)")
    ax.set_ylabel("Mean P(preictal)")
    ax.set_title("Binned mean (5-min bins) with ±1 std")
    ax.set_xlim(max(bin_centers) + 3, -3)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.2)

    # Annotate bin counts
    for bc, bm, bn in zip(bin_centers, bin_means, bin_counts):
        ax.text(bc, bm + 0.06, f"n={bn}", ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "global_temporal_trend.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # === Plot 2: Per-patient trends ===
    n_patients = len(per_patient)
    if n_patients == 0:
        return

    cols = 4
    rows = (n_patients + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n_patients > 1 else [axes]

    for i, (pid, (times, probs)) in enumerate(sorted(per_patient.items())):
        ax = axes[i]
        ax.scatter(times, probs, alpha=0.3, s=10, c="#1976d2")

        # Trend line
        if len(times) > 5:
            z = np.polyfit(times, probs, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(times), max(times), 50)
            ax.plot(x_line, p(x_line), color="#d32f2f", linewidth=2, linestyle="--")
            slope = z[0]
            ax.set_title(f"{pid} (n={len(times)}, slope={slope:.4f}/min)", fontsize=9)
        else:
            ax.set_title(f"{pid} (n={len(times)})", fontsize=9)

        ax.set_xlim(max(times) + 1 if len(times) > 0 else 41, -1)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Min to seizure", fontsize=8)
        ax.set_ylabel("P(preictal)", fontsize=8)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Per-patient: P(preictal) vs time to seizure onset", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "per_patient_temporal_trend.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # === Summary stats ===
    correlation = np.corrcoef(all_times, all_probs)[0, 1]
    print(f"\nGlobal correlation (time_to_seizure vs P(preictal)): {correlation:.4f}")
    if correlation < 0:
        print("  Negative correlation = probability INCREASES as we approach seizure onset (good)")
    else:
        print("  Positive correlation = probability DECREASES as we approach seizure onset (unexpected)")

    print(f"\nPer-patient slopes (negative = increasing toward seizure):")
    for pid, (times, probs) in sorted(per_patient.items()):
        if len(times) > 5:
            slope = np.polyfit(times, probs, 1)[0]
            direction = "increases" if slope < 0 else "flat/decreases"
            print(f"  {pid}: slope={slope:.5f}/min  ({direction})")


if __name__ == "__main__":
    main()
