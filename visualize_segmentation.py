"""Visualize how the segmentation assigns sequences relative to seizures.

Produces a per-patient timeline showing:
- Seizure locations
- Preictal zones
- Interictal buffer (exclusion) zones
- Where each sequence falls and its label/fold assignment
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from data_segmentation_helpers.config import (
    PREICTAL_WINDOW, PREICTAL_ONSET_BUFFER, INTERICTAL_BUFFER,
    SEGMENT_DURATION, SEQUENCE_LENGTH, PATIENTS, BASE_PATH,
)
from data_segmentation_helpers.segmentation import parse_summary_file

OUTPUT_DIR = Path("preprocessing") / "visualizations" / "segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _parse_time_str(time_str):
    """Parse 'HH:MM:SS' to total seconds."""
    parts = time_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def load_patient_timeline(patient_id):
    """Load seizure times and recording duration from summary file."""
    summary_file = f"{BASE_PATH}{patient_id}/{patient_id}-summary.txt"
    raw_seizures, all_files, file_times = parse_summary_file(summary_file)

    # Build file offsets from start/end times (cumulative duration)
    file_offsets = {}
    cumulative = 0
    for fname in all_files:
        ft = file_times.get(fname, {})
        file_offsets[fname] = cumulative
        if "start_time" in ft and "end_time" in ft:
            start_s = _parse_time_str(ft["start_time"])
            end_s = _parse_time_str(ft["end_time"])
            duration = end_s - start_s
            if duration <= 0:
                duration += 86400  # handle midnight crossing
            cumulative += duration
        else:
            cumulative += 3600  # fallback 1hr

    total_duration = cumulative

    seizures = []
    for i, s in enumerate(raw_seizures):
        offset = file_offsets.get(s["file"], 0)
        seizures.append({
            "start": offset + s["start_sec"],
            "end": offset + s["end_sec"],
            "id": i,
        })

    seizures.sort(key=lambda x: x["start"])
    return seizures, total_duration


def build_file_offsets(patient_id):
    """Build a dict mapping filename → global offset in seconds."""
    summary_file = f"{BASE_PATH}{patient_id}/{patient_id}-summary.txt"
    _, all_files, file_times = parse_summary_file(summary_file)

    offsets = {}
    cumulative = 0
    for fname in all_files:
        offsets[fname] = cumulative
        ft = file_times.get(fname, {})
        if "start_time" in ft and "end_time" in ft:
            start_s = _parse_time_str(ft["start_time"])
            end_s = _parse_time_str(ft["end_time"])
            duration = end_s - start_s
            if duration <= 0:
                duration += 86400
            cumulative += duration
        else:
            cumulative += 3600
    return offsets


def plot_patient(patient_id, seizures, total_duration, sequences):
    """Create a timeline plot for one patient."""
    seq_duration = SEGMENT_DURATION * SEQUENCE_LENGTH

    fig, ax = plt.subplots(figsize=(20, 4))

    # Convert to hours for readability
    to_hr = lambda s: s / 3600

    # 1. Draw recording span
    ax.axhspan(-0.5, 2.5, xmin=0, xmax=1, color="#f0f0f0", zorder=0)

    # 2. Draw exclusion buffers (interictal buffer zones)
    for sz in seizures:
        # Pre-seizure buffer
        buf_start = max(0, sz["start"] - INTERICTAL_BUFFER)
        buf_end = max(0, sz["start"] - PREICTAL_WINDOW)
        if buf_end > buf_start:
            ax.axvspan(to_hr(buf_start), to_hr(buf_end), color="#ffe0b2", alpha=0.5, zorder=1)

        # Post-seizure buffer
        post_start = sz["end"]
        post_end = sz["end"] + INTERICTAL_BUFFER
        ax.axvspan(to_hr(post_start), to_hr(post_end), color="#ffe0b2", alpha=0.5, zorder=1)

    # 3. Draw preictal zones
    for sz in seizures:
        pre_start = max(0, sz["start"] - PREICTAL_WINDOW)
        pre_end = sz["start"] - PREICTAL_ONSET_BUFFER
        if pre_end > pre_start:
            ax.axvspan(to_hr(pre_start), to_hr(pre_end), color="#ffcdd2", alpha=0.6, zorder=2)

    # 4. Draw seizures
    for sz in seizures:
        ax.axvspan(to_hr(sz["start"]), to_hr(sz["end"]), color="#d32f2f", alpha=0.9, zorder=5)
        ax.text(to_hr((sz["start"] + sz["end"]) / 2), 2.3, f'sz{sz["id"]}',
                ha="center", va="bottom", fontsize=7, color="#d32f2f", fontweight="bold")

    # 5. Draw sequences as thin markers on the timeline
    file_offsets = build_file_offsets(patient_id)
    preictal_xs = []
    interictal_xs = []
    for seq in sequences:
        start_sec = seq["sequence_start_sec"]
        end_sec = start_sec + seq_duration
        file_offset = file_offsets.get(seq["file"], 0)

        global_start = file_offset + start_sec
        global_end = file_offset + end_sec
        center = to_hr((global_start + global_end) / 2)

        if seq["type"] == "preictal":
            preictal_xs.append(center)
        else:
            interictal_xs.append(center)

    # Plot sequence markers
    if interictal_xs:
        ax.scatter(interictal_xs, [0.5] * len(interictal_xs), c="#4caf50", s=8,
                   marker="|", linewidths=0.8, zorder=4, label=f"Interictal ({len(interictal_xs)})")
    if preictal_xs:
        ax.scatter(preictal_xs, [1.5] * len(preictal_xs), c="#e53935", s=8,
                   marker="|", linewidths=0.8, zorder=4, label=f"Preictal ({len(preictal_xs)})")

    # Formatting
    ax.set_xlim(0, to_hr(total_duration))
    ax.set_ylim(-0.5, 3.0)
    ax.set_xlabel("Time (hours)")
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Interictal", "Preictal"])
    ax.set_title(f"{patient_id}: {len(seizures)} seizures, "
                 f"{len(preictal_xs)} preictal + {len(interictal_xs)} interictal sequences "
                 f"({to_hr(total_duration):.1f} hrs recording)")

    # Legend
    handles = [
        mpatches.Patch(color="#d32f2f", alpha=0.9, label="Seizure"),
        mpatches.Patch(color="#ffcdd2", alpha=0.6, label="Preictal zone"),
        mpatches.Patch(color="#ffe0b2", alpha=0.5, label="Interictal buffer (excluded)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{patient_id}_timeline.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_summary(all_stats):
    """Bar chart summarizing all patients."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    patients = [s["pid"] for s in all_stats]
    preictal = [s["preictal"] for s in all_stats]
    interictal = [s["interictal"] for s in all_stats]
    seizures = [s["seizures"] for s in all_stats]

    x = np.arange(len(patients))
    w = 0.35

    # Sequence counts
    ax = axes[0]
    ax.bar(x - w/2, preictal, w, label="Preictal", color="#e53935")
    ax.bar(x + w/2, interictal, w, label="Interictal", color="#4caf50")
    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sequences")
    ax.set_title("Sequences per Patient")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Seizure counts
    ax = axes[1]
    ax.bar(x, seizures, color="#1976d2")
    ax.set_xticks(x)
    ax.set_xticklabels(patients, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Seizures")
    ax.set_title("Seizures per Patient")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def main():
    json_files = sorted(glob.glob("*_sequences_prediction.json"))
    if not json_files:
        print("No segmentation JSONs found. Run data_segmentation.py first.")
        return

    all_stats = []

    for jf in json_files:
        pid = jf.split("_")[0]
        print(f"Processing {pid}...")

        with open(jf) as f:
            data = json.load(f)
        sequences = data["sequences"]

        try:
            seizures, total_duration = load_patient_timeline(pid)
        except Exception as e:
            print(f"  Skipping {pid}: {e}")
            continue

        plot_patient(pid, seizures, total_duration, sequences)

        n_pre = sum(1 for s in sequences if s["type"] == "preictal")
        n_int = sum(1 for s in sequences if s["type"] == "interictal")
        all_stats.append({"pid": pid, "preictal": n_pre, "interictal": n_int, "seizures": len(seizures)})

    if all_stats:
        plot_summary(all_stats)
        total_pre = sum(s["preictal"] for s in all_stats)
        total_int = sum(s["interictal"] for s in all_stats)
        print(f"\nTotal: {total_pre} preictal + {total_int} interictal = {total_pre + total_int} sequences")


if __name__ == "__main__":
    main()
