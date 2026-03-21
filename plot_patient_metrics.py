"""
Generate per-patient metric plots from test_results.json files.

Produces 6 separate publication-ready figures (horizontal bar charts):
  1. Accuracy per patient
  2. AUC-ROC per patient
  3. Sensitivity (recall) per patient
  4. Specificity per patient
  5. Seizure detection rate per patient
  6. False positive rate per patient

Each figure is sorted by metric value (descending) and styled for an
ECE496 final report.
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Style constants ──────────────────────────────────────────────────────────
BAR_COLOR = "#4878A8"          # Steel blue — prints well in grayscale
MEAN_COLOR = "#C0392B"         # Muted red for mean line
FIG_WIDTH = 7.0                # Inches — fits single-column report layout
BAR_HEIGHT = 0.55              # Bar thickness
FONT_SIZE_TITLE = 12
FONT_SIZE_LABEL = 10
FONT_SIZE_TICK = 9
FONT_SIZE_ANNOT = 8
DPI = 300


def load_patient_results(model_dir="model"):
    """Load test_results.json for all patients that have one."""
    results = {}
    pattern = os.path.join(model_dir, "*/test_results.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        pid = data["patient_id"]
        results[pid] = data
    return results


def compute_specificity(confusion_matrix):
    """Specificity = TN / (TN + FP). Returns None if denominator is 0."""
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    denom = tn + fp
    return tn / denom if denom > 0 else None


def make_vertical_bar_plot(patients, values, title, ylabel, filename,
                           ylim=(0, 1.12), fmt=".2f", invert_sort=False):
    """
    Create a single vertical bar chart figure and save it to disk.

    Parameters
    ----------
    patients : list of str
        Patient IDs (will be sorted by value).
    values : list of float
        Metric values corresponding to each patient.
    title : str
        Figure title (used as caption-friendly label).
    ylabel : str
        Y-axis label.
    filename : str
        Output filename (under result_plots/).
    ylim : tuple or None
        Y-axis limits. None for auto.
    fmt : str
        Format string for value annotations.
    invert_sort : bool
        If True, sort ascending — useful for FP rate where lower is better.
    """
    # Sort patients by metric value (descending by default)
    paired = list(zip(patients, values))
    if invert_sort:
        paired.sort(key=lambda x: x[1])                  # lowest first
    else:
        paired.sort(key=lambda x: x[1], reverse=True)    # highest first
    sorted_patients, sorted_values = zip(*paired)
    sorted_patients = [p.upper() for p in sorted_patients]

    n = len(sorted_patients)
    fig_width = max(5.0, 0.55 * n + 1.5)  # Scale width with patient count
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))

    x = np.arange(n)
    bars = ax.bar(x, sorted_values, width=0.6, color=BAR_COLOR,
                  edgecolor="white", linewidth=0.5)

    # Value annotations above each bar
    y_max = max(sorted_values) if sorted_values else 1.0
    offset = y_max * 0.02
    for bar, v in zip(bars, sorted_values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                f"{v:{fmt}}", ha="center", va="bottom",
                fontsize=FONT_SIZE_ANNOT, color="#333333")

    # Mean line (horizontal)
    mean_val = np.mean(sorted_values)
    ax.axhline(mean_val, color=MEAN_COLOR, linestyle="--", linewidth=1.0,
               alpha=0.8, zorder=0)
    # Place mean label at right edge
    ax.text(n - 0.5, mean_val, f"  Mean: {mean_val:{fmt}}",
            va="bottom", ha="right", fontsize=FONT_SIZE_ANNOT,
            color=MEAN_COLOR, fontstyle="italic")

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_patients, rotation=45, ha="right",
                       fontsize=FONT_SIZE_TICK)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel("Patient", fontsize=FONT_SIZE_LABEL)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=12)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)

    if ylim is not None:
        ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))

    # Minimal gridlines
    ax.grid(axis="y", alpha=0.2, linestyle="-", which="major")
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    output_path = os.path.join("result_plots", filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


def main():
    results = load_patient_results()
    if not results:
        print("No test_results.json files found in model/*/.")
        return

    patients = sorted(results.keys())
    print(f"Found results for {len(patients)} patients: {patients}\n")

    # Extract metrics
    accuracy = [results[p]["test_metrics"]["accuracy"] for p in patients]
    auc = [results[p]["test_metrics"]["auc_roc"] for p in patients]
    sensitivity = [results[p]["test_metrics"]["recall"] for p in patients]
    specificity = [compute_specificity(results[p]["confusion_matrix"])
                   for p in patients]
    seizure_det = [results[p]["seizure_accuracy_metrics"]["per_seizure_accuracy"]
                   for p in patients]
    fp_rate = [results[p]["seizure_accuracy_metrics"]["fp_rate_per_hour"]
               for p in patients]

    # Replace None specificity with 0 for plotting
    specificity = [s if s is not None else 0.0 for s in specificity]

    # ── Generate 6 individual figures ────────────────────────────────────
    print("Generating figures:")

    make_vertical_bar_plot(
        patients, accuracy,
        title="Per-Patient Classification Accuracy",
        ylabel="Accuracy",
        filename="metric_accuracy.png",
    )
    make_vertical_bar_plot(
        patients, auc,
        title="Per-Patient AUC-ROC",
        ylabel="AUC-ROC",
        filename="metric_auc.png",
    )
    make_vertical_bar_plot(
        patients, sensitivity,
        title="Per-Patient Sensitivity (Recall)",
        ylabel="Sensitivity",
        filename="metric_sensitivity.png",
    )
    make_vertical_bar_plot(
        patients, specificity,
        title="Per-Patient Specificity",
        ylabel="Specificity",
        filename="metric_specificity.png",
    )
    make_vertical_bar_plot(
        patients, seizure_det,
        title="Per-Patient Seizure Detection Rate",
        ylabel="Detection Rate",
        filename="metric_seizure_detection.png",
    )
    make_vertical_bar_plot(
        patients, fp_rate,
        title="Per-Patient False Positive Rate",
        ylabel="False Positives per Hour",
        filename="metric_fp_rate.png",
        ylim=None,  # Auto-scale since FP rate is unbounded
        invert_sort=True,  # Lower is better
    )

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'Patient':<10} {'Acc':>6} {'AUC':>6} {'Sens':>6} "
          f"{'Spec':>6} {'Det':>6} {'FP/h':>7}")
    print("-" * 50)
    for i, p in enumerate(patients):
        print(f"{p:<10} {accuracy[i]:>6.2f} {auc[i]:>6.2f} "
              f"{sensitivity[i]:>6.2f} {specificity[i]:>6.2f} "
              f"{seizure_det[i]:>6.2f} {fp_rate[i]:>7.2f}")
    print("-" * 50)
    print(f"{'Mean':<10} {np.mean(accuracy):>6.2f} {np.mean(auc):>6.2f} "
          f"{np.mean(sensitivity):>6.2f} {np.mean(specificity):>6.2f} "
          f"{np.mean(seizure_det):>6.2f} {np.mean(fp_rate):>7.2f}")
    print(f"{'Std':>10} {np.std(accuracy):>6.2f} {np.std(auc):>6.2f} "
          f"{np.std(sensitivity):>6.2f} {np.std(specificity):>6.2f} "
          f"{np.std(seizure_det):>6.2f} {np.std(fp_rate):>7.2f}")


if __name__ == "__main__":
    main()
