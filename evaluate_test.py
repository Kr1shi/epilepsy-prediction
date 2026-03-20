#!/usr/bin/env python3
"""
Evaluate trained model on test dataset

Usage:
    python evaluate_test.py              # Evaluate best model (default)
    python evaluate_test.py --epoch 10   # Evaluate specific epoch
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm
import json
import argparse
import matplotlib.pyplot as plt

# Import from train.py
from train import EEGDataset, ConvTransformerModel, MetricsTracker
from total_accuracy import calculate_per_seizure_accuracy
from data_segmentation_helpers.config import (
    TASK_MODE,
    SEQUENCE_LENGTH,
    SEQUENCE_BATCH_SIZE,
    CONV_EMBEDDING_DIM,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_NUM_HEADS,
    TRANSFORMER_FFN_DIM,
    TRANSFORMER_DROPOUT,
    USE_CLS_TOKEN,
    TRAINING_EPOCHS,
    PATIENTS,
    PATIENT_INDEX,
    get_patient_config,
    SEGMENT_DURATION,
)

# Smoothing parameters
SMOOTHING_T = 20  # Window size for smoothing (heuristic fallback)

# --- Manual Smoothing Override ---
# Set both to integers to force manual smoothing params for all patients.
# Set to None to use per-patient trained params from checkpoint (recommended).
MANUAL_SMOOTHING_WINDOW = None
MANUAL_SMOOTHING_COUNT = None


def get_positive_label():
    """Get positive class label based on task mode"""
    return "preictal" if TASK_MODE == "prediction" else "ictal"


def apply_smoothing(all_predictions, t, x):
    """Apply smoothing to a list of predictions."""
    smoothed_predictions = []
    for i in range(len(all_predictions) - t + 1):
        window = all_predictions[i : i + t]
        if sum(window) >= x:
            smoothed_predictions.append(1)
        else:
            smoothed_predictions.append(0)
    return smoothed_predictions


def _create_model(device):
    """Create a ConvTransformerModel instance."""
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


def evaluate_model(model_path, test_data_path, device):
    """Load trained model and evaluate on test dataset."""
    # Load test dataset
    print(f"Loading test dataset from {test_data_path}...")
    test_dataset = EEGDataset(test_data_path, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=SEQUENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Initialize Conv-Transformer Model
    model = _create_model(device)

    # Load trained weights
    print(f"Loading model checkpoint from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Get config from checkpoint
    config = checkpoint.get("config", {})
    checkpoint_task_mode = config.get("task_mode", TASK_MODE)
    positive_class = config.get("positive_class", get_positive_label())
    loaded_threshold = 0.5  # Fixed natural decision boundary (no per-patient tuning)
    smoothing_window = config.get("smoothing_window")
    smoothing_count = config.get("smoothing_count")
    temperature = 1.0  # No temperature scaling

    print(f"Model loaded from epoch {checkpoint.get('epoch', 'Best')}")
    print(f"Task mode: {checkpoint_task_mode.upper()} ({positive_class} vs interictal)")
    print(f"Using decision threshold: {loaded_threshold:.4f}")
    print(f"Temperature scaling: {temperature:.2f}")
    if smoothing_window:
        print(
            f"Loaded smoothing params: Window={smoothing_window}, Count={smoothing_count}"
        )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Evaluate
    print("\nEvaluating on test set...")
    metrics_tracker = MetricsTracker()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    num_batches = 0

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")

        for x, labels in pbar:
            x, labels = x.to(device), labels.to(device)

            outputs = model(x)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

            probabilities = torch.softmax(outputs / temperature, dim=1)[:, 1]
            predictions = (probabilities >= loaded_threshold).long()

            metrics_tracker.update(predictions, labels, probabilities)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    metrics = metrics_tracker.compute_metrics()
    metrics["loss"] = avg_loss

    # Record the threshold used (from training, no test-set leakage)
    metrics["threshold_used"] = float(loaded_threshold)

    cm = confusion_matrix(all_labels, all_predictions)

    return (
        metrics,
        cm,
        all_labels,
        all_predictions,
        all_probabilities,
        checkpoint_task_mode,
        positive_class,
        smoothing_window,
        smoothing_count,
    )


def _get_ensemble_probs(n_seeds, data_path, patient_id, device, split="test"):
    """Get averaged probabilities from N seed models on a given dataset."""
    dataset = EEGDataset(data_path, split=split)
    loader = DataLoader(
        dataset,
        batch_size=SEQUENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    all_labels = None
    ensemble_probs = []

    for seed_idx in range(n_seeds):
        suffix = f"seed{seed_idx}"
        model_path = Path("model") / suffix / patient_id / "best_model.pth"
        if not model_path.exists():
            continue

        model = _create_model(device)
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        seed_probs = []
        seed_labels = []
        with torch.no_grad():
            for x, labels in loader:
                x = x.to(device)
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                seed_probs.extend(probs.cpu().numpy())
                seed_labels.extend(labels.numpy())

        ensemble_probs.append(np.array(seed_probs))
        if all_labels is None:
            all_labels = np.array(seed_labels)
        if split == "test":
            print(f"  {suffix}: loaded (epoch {checkpoint.get('epoch', '?')})")

    if not ensemble_probs:
        return None, None, 0

    avg_probs = np.mean(ensemble_probs, axis=0)
    return avg_probs, all_labels, len(ensemble_probs)


def tune_ensemble_threshold(n_seeds, device):
    """Tune threshold on combined val set using ensemble-averaged probabilities."""
    print("Tuning ensemble threshold on combined validation set...")
    all_val_probs = []
    all_val_labels = []

    for patient_id in PATIENTS:
        val_path = Path("preprocessing") / "data" / patient_id / "val_dataset.h5"
        if not val_path.exists():
            continue
        probs, labels, n_models = _get_ensemble_probs(
            n_seeds, str(val_path), patient_id, device, split=f"val ({patient_id})"
        )
        if probs is not None:
            all_val_probs.extend(probs)
            all_val_labels.extend(labels)

    all_val_probs = np.array(all_val_probs)
    all_val_labels = np.array(all_val_labels)

    if len(np.unique(all_val_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(all_val_labels, all_val_probs)
        j_scores = tpr - fpr
        optimal_threshold = float(thresholds[np.argmax(j_scores)])
    else:
        optimal_threshold = 0.5

    val_auc = roc_auc_score(all_val_labels, all_val_probs)
    print(f"  Val samples: {len(all_val_labels)}")
    print(f"  Val AUC: {val_auc:.4f}")
    print(f"  Optimal threshold (Youden's J): {optimal_threshold:.4f}")
    return optimal_threshold


def evaluate_ensemble(n_seeds, test_data_path, patient_id, device, threshold=0.5):
    """Evaluate ensemble by averaging probabilities from N seed models."""
    print(f"Loading test dataset from {test_data_path}...")

    avg_probs, all_labels, n_models = _get_ensemble_probs(
        n_seeds, test_data_path, patient_id, device, split="test"
    )

    if avg_probs is None:
        print(f"  No ensemble models found for {patient_id}")
        return None

    print(f"  Ensemble size: {n_models} models")

    predictions = (avg_probs >= threshold).astype(int)

    # Compute metrics
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, avg_probs)
    else:
        auc = 0.5

    metrics = {
        "accuracy": accuracy_score(all_labels, predictions),
        "precision": precision_score(all_labels, predictions, zero_division=0),
        "recall": recall_score(all_labels, predictions, zero_division=0),
        "f1": f1_score(all_labels, predictions, zero_division=0),
        "auc_roc": auc,
        "threshold_used": threshold,
        "ensemble_size": n_models,
    }

    cm = confusion_matrix(all_labels, predictions)
    positive_class = get_positive_label()

    # Use smoothing params from first seed's checkpoint
    first_model_path = Path("model") / "seed0" / patient_id / "best_model.pth"
    ckpt = torch.load(first_model_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    smoothing_window = config.get("smoothing_window")
    smoothing_count = config.get("smoothing_count")

    return (
        metrics,
        cm,
        all_labels.tolist(),
        predictions.tolist(),
        avg_probs.tolist(),
        TASK_MODE,
        positive_class,
        smoothing_window,
        smoothing_count,
    )


def plot_preictal_dynamics(model, test_dataset, device, patient_id, threshold=0.5, temperature=1.0):
    """Plots the model's output probability for each distinct seizure event in the test set."""
    print("\nGenerating per-seizure dynamics plots...")

    output_dir = Path("result_plots") / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)

    probabilities = []
    true_labels = []

    loader = DataLoader(test_dataset, batch_size=SEQUENCE_BATCH_SIZE, shuffle=False)

    model.eval()
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs / temperature, dim=1)[:, 1]
            probabilities.extend(probs.cpu().numpy())
            true_labels.extend(labels.numpy())

    # Retrieve Metadata for Sorting
    import h5py

    try:
        with h5py.File(test_dataset.h5_file_path, "r") as f:
            filenames = [n.decode("utf-8") for n in f["segment_info/file_names"][:]]
            start_times = f["segment_info/start_times"][:]

            if len(filenames) != len(probabilities):
                print(
                    f"  Metadata length mismatch! Data: {len(probabilities)}, Meta: {len(filenames)}"
                )
                sorted_indices = range(len(probabilities))
            else:
                meta_list = [
                    (i, filenames[i], start_times[i]) for i in range(len(filenames))
                ]
                meta_list.sort(key=lambda x: (x[1], x[2]))
                sorted_indices = [x[0] for x in meta_list]

    except Exception as e:
        print(f"  Could not load metadata for sorting: {e}")
        sorted_indices = range(len(probabilities))

    probs = np.array(probabilities)[sorted_indices]
    labels = np.array(true_labels)[sorted_indices]

    # Identify Seizure Events
    is_preictal = (labels == 1).astype(int)
    diff = np.diff(np.concatenate(([0], is_preictal, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        print("No preictal (seizure) events found in test set to plot.")
        return

    print(f"Found {len(starts)} contiguous seizure events (after sorting).")

    buffer_steps = 50

    for i, (start, end) in enumerate(zip(starts, ends)):
        plot_start = max(0, start - buffer_steps)
        plot_end = min(len(probs), end + int(buffer_steps / 4))

        segment_probs = probs[plot_start:plot_end]
        x_axis = np.arange(plot_start, plot_end)

        plt.figure(figsize=(12, 6))
        plt.plot(
            x_axis,
            segment_probs,
            label="Seizure Probability",
            color="blue",
            linewidth=1.5,
        )
        plt.axhline(
            y=threshold,
            color="black",
            linestyle="--",
            label=f"Threshold ({threshold:.2f})",
        )
        plt.axvspan(start, end, color="red", alpha=0.2, label="True Preictal Period")

        plt.title(f"Seizure {i+1} Dynamics (Patient {patient_id})")
        plt.xlabel("Sequence Index (Chronological)")
        plt.ylabel("Probability")
        plt.ylim(-0.05, 1.05)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        filename = f"seizure_{i+1:02d}.png"
        save_path = output_dir / filename
        plt.savefig(save_path)
        plt.close()
        print(f"  Saved {save_path}")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on test dataset"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch number to evaluate. If not specified, loads best_model.pth",
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Evaluate a specific patient by ID (e.g., --patient chb07)",
    )
    parser.add_argument(
        "--ensemble",
        type=int,
        default=None,
        help="Evaluate ensemble of N seed models by averaging probabilities. "
             "Example: --ensemble 5",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Evaluate pretrained encoder directly (no fine-tuned model needed). "
             "Example: --pretrained model/pretrained_encoder.pth",
    )
    args = parser.parse_args()

    n_patients = len(PATIENTS)

    if args.patient:
        if args.patient not in PATIENTS:
            print(f"Error: {args.patient} not in PATIENTS list")
            return
        patient_idx = PATIENTS.index(args.patient)
        patients_to_process = [patient_idx]
        print("=" * 60)
        print(f"EVALUATION: Patient {patient_idx} ({args.patient})")
        print("=" * 60)
    elif PATIENT_INDEX is None:
        patients_to_process = list(range(n_patients))
        print("=" * 60)
        print(f"EVALUATION: ALL {n_patients} PATIENTS")
        print("=" * 60)
    else:
        patients_to_process = [PATIENT_INDEX]
        patient_cfg = get_patient_config(PATIENT_INDEX)
        print("=" * 60)
        print(f"EVALUATION: Patient {PATIENT_INDEX} ({patient_cfg['patient_id']})")
        print("=" * 60)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Tune ensemble threshold on val set (before per-patient evaluation)
    ensemble_threshold = 0.5
    if args.ensemble:
        ensemble_threshold = tune_ensemble_threshold(args.ensemble, device)

    batch_results = {}

    for current_idx in patients_to_process:
        patient_config = get_patient_config(current_idx)
        current_output_prefix = patient_config["output_prefix"]
        patient_id = patient_config["patient_id"]

        print(f"\n{'='*60}")
        print(f"PATIENT {current_idx}/{n_patients-1}: {patient_id}")
        print(f"{'='*60}")

        try:
            dataset_dir = Path("preprocessing") / "data" / current_output_prefix
            test_data_path = dataset_dir / "test_dataset.h5"

            if not test_data_path.exists():
                print(f"  Test dataset not found: {test_data_path}")
                continue

            if args.pretrained:
                # Pretrained-only evaluation: load raw state dict
                print(f"Evaluating PRETRAINED encoder: {args.pretrained}")
                print(f"Dataset prefix: {current_output_prefix}")

                model = _create_model(device)
                model.load_state_dict(
                    torch.load(args.pretrained, map_location=device, weights_only=False)
                )
                model.to(device)
                model.eval()

                # Load pretrained config for threshold/smoothing
                pretrained_config_path = Path(args.pretrained).parent / "pretrained_config.json"
                if pretrained_config_path.exists():
                    with open(pretrained_config_path) as f:
                        pt_config = json.load(f)
                    loaded_threshold = pt_config.get("optimal_threshold", 0.5)
                    temperature = pt_config.get("temperature", 1.0)
                    ckpt_window = pt_config.get("smoothing_window")
                    ckpt_count = pt_config.get("smoothing_count")
                else:
                    loaded_threshold = 0.5
                    temperature = 1.0
                    ckpt_window = None
                    ckpt_count = None

                print(f"Threshold: {loaded_threshold:.4f}, Temperature: {temperature:.2f}")

                test_dataset = EEGDataset(test_data_path, split="test")
                test_loader = DataLoader(
                    test_dataset, batch_size=SEQUENCE_BATCH_SIZE, shuffle=False, num_workers=0
                )

                all_predictions = []
                all_labels = []
                all_probabilities = []
                total_loss = 0.0
                criterion = nn.CrossEntropyLoss()

                with torch.no_grad():
                    for x, labels in tqdm(test_loader, desc="Testing"):
                        x, labels = x.to(device), labels.to(device)
                        outputs = model(x)
                        loss = criterion(outputs, labels)
                        total_loss += loss.item()
                        probabilities = torch.softmax(outputs / temperature, dim=1)[:, 1]
                        preds = (probabilities >= loaded_threshold).long()
                        all_predictions.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        all_probabilities.extend(probabilities.cpu().numpy())

                true_labels = all_labels
                predictions = all_predictions
                probabilities = all_probabilities

                labels_np = np.array(all_labels)
                probs_np = np.array(all_probabilities)
                preds_np = np.array(all_predictions)

                if len(np.unique(labels_np)) > 1:
                    auc = roc_auc_score(labels_np, probs_np)
                else:
                    auc = 0.5

                metrics = {
                    "accuracy": accuracy_score(labels_np, preds_np),
                    "precision": precision_score(labels_np, preds_np, zero_division=0),
                    "recall": recall_score(labels_np, preds_np, zero_division=0),
                    "f1": f1_score(labels_np, preds_np, zero_division=0),
                    "auc_roc": auc,
                    "loss": total_loss / max(len(test_loader), 1),
                    "threshold_used": loaded_threshold,
                }
                cm = confusion_matrix(labels_np, preds_np)
                checkpoint_task_mode = TASK_MODE
                positive_class = get_positive_label()

            elif args.ensemble:
                # Ensemble evaluation: average probabilities from N seed models
                print(f"Evaluating ENSEMBLE ({args.ensemble} seeds)")
                print(f"Dataset prefix: {current_output_prefix}")

                result = evaluate_ensemble(args.ensemble, test_data_path, patient_id, device, threshold=ensemble_threshold)
                if result is None:
                    continue

                (
                    metrics,
                    cm,
                    true_labels,
                    predictions,
                    probabilities,
                    checkpoint_task_mode,
                    positive_class,
                    ckpt_window,
                    ckpt_count,
                ) = result
            else:
                if args.epoch is not None:
                    model_filename = f"epoch_{args.epoch:03d}.pth"
                else:
                    model_filename = "best_model.pth"

                model_path = Path(f"model/{current_output_prefix}/{model_filename}")

                if not model_path.exists():
                    last_epoch = Path(
                        f"model/{current_output_prefix}/epoch_{TRAINING_EPOCHS:03d}.pth"
                    )
                    if args.epoch is None and last_epoch.exists():
                        print(
                            f"  'best_model.pth' not found. Falling back to last epoch: {last_epoch}"
                        )
                        model_path = last_epoch
                    else:
                        print(f"  Model not found: {model_path}")
                        continue

                print(f"Evaluating model: {model_path.name}")
                print(f"Dataset prefix: {current_output_prefix}")

                (
                    metrics,
                    cm,
                    true_labels,
                    predictions,
                    probabilities,
                    checkpoint_task_mode,
                    positive_class,
                    ckpt_window,
                    ckpt_count,
                ) = evaluate_model(model_path, test_data_path, device)

            # Print results
            print("\n" + "=" * 60)
            print("PATIENT TEST RESULTS")
            print("=" * 60)
            if "loss" in metrics:
                print(f"Loss:      {metrics['loss']:.4f}")
            print(
                f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)"
            )
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1']:.4f}")
            print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")

            print(f"\nThreshold used (from training): {metrics['threshold_used']:.4f}")

            print("\nConfusion Matrix:")
            print("                Predicted")
            print(f"                Interictal  {positive_class.capitalize()}")
            print(f"Actual Interictal    {cm[0,0]:6d}    {cm[0,1]:6d}")
            if cm.shape[0] > 1:
                print(
                    f"       {positive_class.capitalize():9s}   {cm[1,0]:6d}    {cm[1,1]:6d}"
                )

            patient_results = {
                "patient_id": patient_id,
                "task_mode": checkpoint_task_mode,
                "test_metrics": metrics,
                "confusion_matrix": cm.tolist(),
            }
            if args.pretrained:
                patient_results["model_path"] = args.pretrained
            elif not args.ensemble:
                patient_results["model_path"] = str(model_path)

            # Apply smoothing
            if (
                MANUAL_SMOOTHING_WINDOW is not None
                and MANUAL_SMOOTHING_COUNT is not None
            ):
                window_size = MANUAL_SMOOTHING_WINDOW
                threshold_x = MANUAL_SMOOTHING_COUNT
                print(f"Applying smoothing using MANUAL parameters:")
                print(f"  - Window Size: {window_size}")
                print(f"  - Threshold Count: {threshold_x}")
            elif ckpt_window is not None and ckpt_count is not None:
                window_size = ckpt_window
                threshold_x = ckpt_count
                print(f"\nApplying smoothing using TRAINED parameters:")
                print(f"  - Window Size: {window_size}")
                print(f"  - Threshold Count: {threshold_x}")
            elif SMOOTHING_T > 1:
                window_size = SMOOTHING_T
                model_precision = metrics["precision"]
                target_count = int(round(model_precision * window_size))
                # Cap at 50% of window to avoid near-unanimity requirement
                threshold_x = max(1, min(target_count - 1, window_size // 2))
                print(f"\nApplying smoothing using HEURISTIC parameters (fallback):")
                print(f"  - Window Size: {window_size}")
                print(
                    f"  - Threshold Count: {threshold_x} (derived from precision {model_precision:.4f})"
                )
            else:
                window_size = 1
                threshold_x = 1

            # Cap window size to test set length, scaling count proportionally
            if window_size > len(predictions):
                original_window = window_size
                window_size = max(1, len(predictions))
                threshold_x = max(1, round(threshold_x * window_size / original_window))

            if window_size > 1:
                smoothed_predictions = apply_smoothing(
                    predictions, window_size, threshold_x
                )
                smoothed_true_labels = true_labels[window_size - 1 :]

                smoothed_metrics = {
                    "accuracy": accuracy_score(
                        smoothed_true_labels, smoothed_predictions
                    ),
                    "precision": precision_score(
                        smoothed_true_labels, smoothed_predictions, zero_division=0
                    ),
                    "recall": recall_score(
                        smoothed_true_labels, smoothed_predictions, zero_division=0
                    ),
                    "f1": f1_score(
                        smoothed_true_labels, smoothed_predictions, zero_division=0
                    ),
                }
                patient_results["smoothed_metrics"] = smoothed_metrics

                print("\n" + "=" * 60)
                print("SMOOTHED PREDICTIONS RESULTS")
                print("=" * 60)
                print(
                    f"Smoothed_Accuracy:  {smoothed_metrics['accuracy']:.4f} ({smoothed_metrics['accuracy']*100:.2f}%)"
                )
                print(f"Smoothed_Precision: {smoothed_metrics['precision']:.4f}")
                print(f"Smoothed_Recall:    {smoothed_metrics['recall']:.4f}")
                print(f"Smoothed_F1 Score:  {smoothed_metrics['f1']:.4f}")

            # Per-seizure accuracy (runs for all patients, including window_size=1)
            test_dataset = EEGDataset(test_data_path, split="test")

            seizure_accuracy_metrics = calculate_per_seizure_accuracy(
                true_labels,
                predictions,
                test_dataset,
                SEGMENT_DURATION * SEQUENCE_LENGTH,
                window_size,
                threshold_x,
            )
            patient_results["seizure_accuracy_metrics"] = seizure_accuracy_metrics

            print("\n" + "=" * 60)
            print("PER-SEIZURE ACCURACY RESULTS")
            print("=" * 60)
            print(f"Total Seizures: {seizure_accuracy_metrics['total_seizures']}")
            print(
                f"Per-Seizure Accuracy: {seizure_accuracy_metrics['per_seizure_accuracy']:.4f} "
                f"({seizure_accuracy_metrics['correctly_predicted_seizures']}/{seizure_accuracy_metrics['total_seizures']})"
            )
            print(
                f"False Positive Rate: {seizure_accuracy_metrics['fp_rate_per_hour']:.4f} per hour"
            )

            if args.ensemble:
                results_dir = Path("model") / "ensemble"
                results_dir.mkdir(parents=True, exist_ok=True)
                patient_results_path = results_dir / f"{current_output_prefix}_test_results.json"
            else:
                patient_results_path = Path(
                    f"model/{current_output_prefix}/test_results.json"
                )
            with open(patient_results_path, "w") as f:
                json.dump(patient_results, f, indent=2)

            print(f"\n  Results saved to {patient_results_path}")

            # Generate Dynamics Plot
            if args.pretrained:
                plot_model_path = Path(args.pretrained)
            elif args.ensemble:
                plot_model_path = Path("model") / "seed0" / patient_id / "best_model.pth"
            else:
                plot_model_path = model_path

            if args.pretrained:
                # Model already loaded above
                pass
            elif plot_model_path.exists():
                model = _create_model(device)
                try:
                    checkpoint = torch.load(
                        plot_model_path, map_location=device, weights_only=False
                    )
                except TypeError:
                    checkpoint = torch.load(plot_model_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(device)

                if "test_dataset" not in locals():
                    test_dataset = EEGDataset(test_data_path, split="test")

                plot_preictal_dynamics(
                    model=model,
                    test_dataset=test_dataset,
                    device=device,
                    patient_id=patient_id,
                    threshold=0.5,
                    temperature=1.0,
                )

            batch_results[current_idx] = {
                "patient_id": patient_id,
                "metrics": metrics,
            }
            if "smoothed_metrics" in patient_results:
                batch_results[current_idx]["smoothed_metrics"] = patient_results[
                    "smoothed_metrics"
                ]
            if "seizure_accuracy_metrics" in patient_results:
                batch_results[current_idx]["seizure_accuracy_metrics"] = (
                    patient_results["seizure_accuracy_metrics"]
                )

        except Exception as e:
            print(f"  Error evaluating patient {patient_id}: {e}")
            import traceback

            traceback.print_exc()

    # Save batch results summary
    if PATIENT_INDEX is None and batch_results:
        print("\n" + "=" * 60)
        print("BATCH EVALUATION SUMMARY")
        print("=" * 60)

        accuracies = [res["metrics"]["accuracy"] for res in batch_results.values()]
        auc_rocs = [res["metrics"]["auc_roc"] for res in batch_results.values()]

        print(f"Patients evaluated: {len(batch_results)}/{len(patients_to_process)}")
        print(f"Mean Accuracy: {np.mean(accuracies):.4f} (+/-{np.std(accuracies):.4f})")
        print(f"Mean AUC-ROC:  {np.mean(auc_rocs):.4f} (+/-{np.std(auc_rocs):.4f})")

        smoothed_accuracies = [
            res["smoothed_metrics"]["accuracy"]
            for res in batch_results.values()
            if "smoothed_metrics" in res
        ]
        if smoothed_accuracies:
            print(
                f"Mean Smoothed Accuracy: {np.mean(smoothed_accuracies):.4f} (+/-{np.std(smoothed_accuracies):.4f})"
            )

        per_seizure_accuracies = [
            res["seizure_accuracy_metrics"]["per_seizure_accuracy"]
            for res in batch_results.values()
            if "seizure_accuracy_metrics" in res
            and res["seizure_accuracy_metrics"]["total_seizures"] > 0
        ]
        if per_seizure_accuracies:
            print(
                f"Mean Per-Seizure Accuracy: {np.mean(per_seizure_accuracies):.4f} (+/-{np.std(per_seizure_accuracies):.4f})"
            )

        fp_rates = [
            res["seizure_accuracy_metrics"]["fp_rate_per_hour"]
            for res in batch_results.values()
            if "seizure_accuracy_metrics" in res
        ]
        if fp_rates:
            print(
                f"Mean False Positive Rate: {np.mean(fp_rates):.4f} per hour (+/-{np.std(fp_rates):.4f})"
            )

        batch_summary = {
            "total_patients": n_patients,
            "patient_results": batch_results,
            "mean_accuracy": float(np.mean(accuracies)),
            "mean_auc": float(np.mean(auc_rocs)),
        }
        if smoothed_accuracies:
            batch_summary["mean_smoothed_accuracy"] = float(
                np.mean(smoothed_accuracies)
            )
        if per_seizure_accuracies:
            batch_summary["mean_per_seizure_accuracy"] = float(
                np.mean(per_seizure_accuracies)
            )
        if fp_rates:
            batch_summary["mean_fp_rate_per_hour"] = float(np.mean(fp_rates))

        with open("model/batch_test_results.json", "w") as f:
            json.dump(batch_summary, f, indent=2)

        print(f"\n  Batch results saved to model/batch_test_results.json")


if __name__ == "__main__":
    main()
