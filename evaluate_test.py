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
from train import EEGDataset, CNN_LSTM_Hybrid_Dual, MetricsTracker
from total_accuracy import calculate_per_seizure_accuracy
from data_segmentation_helpers.config import (
    TASK_MODE,
    SEQUENCE_LENGTH,
    SEQUENCE_BATCH_SIZE,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
    TRAINING_EPOCHS,
    PATIENTS,
    PATIENT_INDEX,
    get_patient_config,
    SEGMENT_DURATION,
)

# Smoothing parameters
SMOOTHING_T = 20  # Window size for smoothing (heuristic fallback)

# --- Manual Smoothing Override ---
# Set these values to manually control smoothing.
# If both are integers, they will override any other settings.
# Set to None to use parameters from the model checkpoint or heuristics.
MANUAL_SMOOTHING_WINDOW = 17  # e.g., 15
MANUAL_SMOOTHING_COUNT =  15  # e.g., 5


def get_positive_label():
    """Get positive class label based on task mode"""
    return "preictal" if TASK_MODE == "prediction" else "ictal"


def apply_smoothing(all_predictions, t, x):
    """
    Apply smoothing to a list of predictions.

    Args:
        all_predictions: List of predictions.
        t: Window size for smoothing.
        x: Minimum number of positive predictions in the window.

    Returns:
        List of smoothed predictions.
    """
    smoothed_predictions = []
    for i in range(len(all_predictions) - t + 1):
        window = all_predictions[i : i + t]
        if sum(window) >= x:
            smoothed_predictions.append(1)
        else:
            smoothed_predictions.append(0)
    return smoothed_predictions


def evaluate_model(model_path, test_data_path, device):
    """
    Load trained model and evaluate on test dataset

    Args:
        model_path: Path to saved model checkpoint (.pth file)
        test_data_path: Path to test dataset HDF5 file
        device: torch device to use

    Returns:
        tuple: (metrics, cm, all_labels, all_predictions, all_probabilities,
                checkpoint_task_mode, positive_class, smoothing_window, smoothing_count)
    """
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

    # Initialize Dual-Stream Model
    model = CNN_LSTM_Hybrid_Dual(
        num_input_channels=18,
        num_classes=2,
        sequence_length=SEQUENCE_LENGTH,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        lstm_num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
    )

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
    loaded_threshold = config.get("optimal_threshold", 0.5)
    smoothing_window = config.get("smoothing_window")
    smoothing_count = config.get("smoothing_count")

    print(f"Model loaded from epoch {checkpoint.get('epoch', 'Best')}")
    print(f"Task mode: {checkpoint_task_mode.upper()} ({positive_class} vs interictal)")
    print(f"Using decision threshold: {loaded_threshold:.4f}")
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

        for x_phase, x_amp, labels in pbar:
            x_phase, x_amp, labels = (
                x_phase.to(device),
                x_amp.to(device),
                labels.to(device),
            )

            outputs = model(x_phase, x_amp)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

            # Get probabilities for positive class
            probabilities = torch.softmax(outputs, dim=1)[:, 1]

            # Use loaded threshold for hard predictions
            predictions = (probabilities >= loaded_threshold).long()

            metrics_tracker.update(predictions, labels, probabilities)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    metrics = metrics_tracker.compute_metrics()
    metrics["loss"] = avg_loss

    # Compute optimal threshold for this specific set (for analysis only)
    labels_np = np.array(all_labels)
    probs_np = np.array(all_probabilities)
    if len(np.unique(labels_np)) > 1:
        fpr, tpr, thresholds = roc_curve(labels_np, probs_np)
        j_scores = tpr - fpr
        best_threshold = thresholds[np.argmax(j_scores)]
    else:
        best_threshold = 0.5
    metrics["test_set_optimal_threshold"] = float(best_threshold)

    # Compute confusion matrix
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


def plot_preictal_dynamics(model, test_dataset, device, patient_id, threshold=0.5):
    """
    Plots the model's output probability for each distinct seizure event in the test set.
    Saves plots to 'result_plots/{patient_id}/'.
    SORTING ENABLED: Reconstructs chronological order using HDF5 metadata.
    """
    print("\nGenerating per-seizure dynamics plots...")
    
    # 1. Setup Output Directory
    output_dir = Path("result_plots") / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Get Predictions and Labels
    probabilities = []
    true_labels = []
    
    loader = DataLoader(test_dataset, batch_size=SEQUENCE_BATCH_SIZE, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        for x_phase, x_amp, labels in loader:
            x_phase = x_phase.to(device)
            x_amp = x_amp.to(device)
            
            outputs = model(x_phase, x_amp)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            probabilities.extend(probs.cpu().numpy())
            true_labels.extend(labels.numpy())
            
    # 3. Retrieve Metadata for Sorting
    import h5py
    try:
        with h5py.File(test_dataset.h5_file_path, "r") as f:
            # Decode bytes to strings for filenames
            filenames = [n.decode('utf-8') for n in f["segment_info/file_names"][:]]
            start_times = f["segment_info/start_times"][:]
            
            # Verify lengths match
            if len(filenames) != len(probabilities):
                print(f"⚠️ Metadata length mismatch! Data: {len(probabilities)}, Meta: {len(filenames)}")
                # Fallback to unsorted if mismatch (shouldn't happen)
                sorted_indices = range(len(probabilities))
            else:
                # Create a list of tuples (index, filename, start_time)
                meta_list = []
                for i in range(len(filenames)):
                    meta_list.append((i, filenames[i], start_times[i]))
                
                # Sort by filename, then start_time
                # This assumes filenames (e.g., chb01_03.edf) sort chronologically, which is true for CHB-MIT
                meta_list.sort(key=lambda x: (x[1], x[2]))
                
                sorted_indices = [x[0] for x in meta_list]
                
    except Exception as e:
        print(f"⚠️ Could not load metadata for sorting: {e}")
        sorted_indices = range(len(probabilities))

    # Apply Sort
    probs = np.array(probabilities)[sorted_indices]
    labels = np.array(true_labels)[sorted_indices]
    
    # 4. Identify Seizure Events (Contiguous blocks of label=1)
    # We find transitions from 0 to 1 (start) and 1 to 0 (end)
    is_preictal = (labels == 1).astype(int)
    diff = np.diff(np.concatenate(([0], is_preictal, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(starts) == 0:
        print("No preictal (seizure) events found in test set to plot.")
        return

    print(f"Found {len(starts)} contiguous seizure events (after sorting).")
    
    # 5. Plot each event
    buffer_steps = 50 # How much interictal context to show before/after
    
    for i, (start, end) in enumerate(zip(starts, ends)):
        # Define window with buffer
        plot_start = max(0, start - buffer_steps)
        plot_end = min(len(probs), end + int(buffer_steps/4)) # smaller buffer after
        
        segment_probs = probs[plot_start:plot_end]
        # segment_labels = labels[plot_start:plot_end] # Unused for plotting logic, used implicit logic
        x_axis = np.arange(plot_start, plot_end)
        
        plt.figure(figsize=(12, 6))
        
        # Plot probabilities
        plt.plot(x_axis, segment_probs, label="Seizure Probability", color="blue", linewidth=1.5)
        
        # Threshold line
        plt.axhline(y=threshold, color="black", linestyle="--", label=f"Threshold ({threshold:.2f})")
        
        # Highlight Preictal Region
        # We fill only where label is 1. Since we know start/end, we can just highlight that block.
        # This highlights the Ground Truth preictal period
        plt.axvspan(start, end, color='red', alpha=0.2, label="True Preictal Period")
        
        plt.title(f"Seizure {i+1} Dynamics (Patient {patient_id})")
        plt.xlabel("Sequence Index (Chronological)")
        plt.ylabel("Probability")
        plt.ylim(-0.05, 1.05)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save
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
    args = parser.parse_args()

    n_patients = len(PATIENTS)

    # Determine which patients to process
    if PATIENT_INDEX is None:
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

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Store results for all folds
    batch_results = {}

    # Evaluate each fold
    for current_idx in patients_to_process:
        patient_config = get_patient_config(current_idx)
        current_output_prefix = patient_config["output_prefix"]
        patient_id = patient_config["patient_id"]

        print(f"\n{'='*60}")
        print(f"PATIENT {current_idx}/{n_patients-1}: {patient_id}")
        print(f"{ '='*60}")

        try:
            # Determine model path
            if args.epoch is not None:
                model_filename = f"epoch_{args.epoch:03d}.pth"
            else:
                model_filename = "best_model.pth"

            model_path = Path(f"model/{current_output_prefix}/{model_filename}")
            dataset_dir = Path("preprocessing") / "data" / current_output_prefix
            test_data_path = dataset_dir / "test_dataset.h5"

            # Check if files exist
            if not model_path.exists():
                last_epoch = Path(
                    f"model/{current_output_prefix}/epoch_{TRAINING_EPOCHS:03d}.pth"
                )
                if args.epoch is None and last_epoch.exists():
                    print(
                        f"⚠️ 'best_model.pth' not found. Falling back to last epoch: {last_epoch}"
                    )
                    model_path = last_epoch
                else:
                    print(f"❌ Model not found: {model_path}")
                    continue

            if not test_data_path.exists():
                print(f"❌ Test dataset not found: {test_data_path}")
                continue

            print(f"Evaluating model: {model_path.name}")
            print(f"Dataset prefix: {current_output_prefix}")

            # Initialize and load model for plotting (needed separately for the plot function)
            # Or we can just reuse the one loaded inside evaluate_model if we refactor,
            # but to keep it clean we will load it again or pass it out.
            # evaluate_model doesn't return the model object.
            # We will initialize a fresh one for plotting.
            
            # Note: We need to load the checkpoint to get the threshold first
            # But evaluate_model already did that.
            
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
            print(f"Loss:      {metrics['loss']:.4f}")
            print(
                f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)"
            )
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1']:.4f}")
            print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")

            print(f"\nTest Set Analysis:")
            print(
                f"Test Set Optimal Threshold: {metrics['test_set_optimal_threshold']:.4f}"
            )

            print("\nConfusion Matrix:")
            print("                Predicted")
            print(f"                Interictal  {positive_class.capitalize()}")
            print(f"Actual Interictal    {cm[0,0]:6d}    {cm[0,1]:6d}")
            if cm.shape[0] > 1:
                print(
                    f"       {positive_class.capitalize():9s}   {cm[1,0]:6d}    {cm[1,1]:6d}"
                )

            # Save fold-specific results
            patient_results = {
                "patient_id": patient_id,
                "task_mode": checkpoint_task_mode,
                "test_metrics": metrics,
                "confusion_matrix": cm.tolist(),
                "model_path": str(model_path),
            }

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
                # Fallback to heuristic: threshold slightly less than precision
                window_size = SMOOTHING_T
                model_precision = metrics["precision"]
                target_count = int(round(model_precision * window_size))
                threshold_x = max(1, min(target_count - 1, window_size))
                print(f"\nApplying smoothing using HEURISTIC parameters (fallback):")
                print(f"  - Window Size: {window_size}")
                print(
                    f"  - Threshold Count: {threshold_x} (derived from precision {model_precision:.4f})"
                )
            else:
                window_size = 1
                threshold_x = 1

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

                # Load dataset for chronological analysis
                test_dataset = EEGDataset(test_data_path, split="test")

                # Calculate per-seizure accuracy
                seizure_accuracy_metrics = calculate_per_seizure_accuracy(
                    true_labels,
                    predictions,
                    test_dataset,
                    SEGMENT_DURATION,
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
                print(f"False Positive Rate: {seizure_accuracy_metrics['fp_rate_per_hour']:.4f} per hour")


            patient_results_path = Path(
                f"model/{current_output_prefix}/test_results.json"
            )
            with open(patient_results_path, "w") as f:
                json.dump(patient_results, f, indent=2)

            print(f"\n✅ Results saved to {patient_results_path}")
            
            # --- Generate Dynamics Plot ---
            # Load model again for plotting since we didn't pass it out
            model = CNN_LSTM_Hybrid_Dual(
                num_input_channels=18,
                num_classes=2,
                sequence_length=SEQUENCE_LENGTH,
                lstm_hidden_dim=LSTM_HIDDEN_DIM,
                lstm_num_layers=LSTM_NUM_LAYERS,
                dropout=LSTM_DROPOUT,
            )
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            
            # We also need the threshold used by evaluate_model, which it extracted from the checkpoint
            # We can re-extract it here
            loaded_threshold = checkpoint.get("config", {}).get("optimal_threshold", 0.5)
            
            # Load dataset for plotting
            if 'test_dataset' not in locals():
                test_dataset = EEGDataset(test_data_path, split="test")
            
            plot_preictal_dynamics(
                model=model, 
                test_dataset=test_dataset, 
                device=device, 
                patient_id=patient_id,
                threshold=loaded_threshold
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
                batch_results[current_idx]["seizure_accuracy_metrics"] = patient_results[
                    "seizure_accuracy_metrics"
                ]

        except Exception as e:
            print(f"❌ Error evaluating patient {patient_id}: {e}")
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
        print(f"Mean Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
        print(f"Mean AUC-ROC:  {np.mean(auc_rocs):.4f} (±{np.std(auc_rocs):.4f})")

        smoothed_accuracies = [
            res["smoothed_metrics"]["accuracy"]
            for res in batch_results.values()
            if "smoothed_metrics" in res
        ]
        if smoothed_accuracies:
            print(
                f"Mean Smoothed Accuracy: {np.mean(smoothed_accuracies):.4f} (±{np.std(smoothed_accuracies):.4f})"
            )
        
        per_seizure_accuracies = [
            res["seizure_accuracy_metrics"]["per_seizure_accuracy"]
            for res in batch_results.values()
            if "seizure_accuracy_metrics" in res and res["seizure_accuracy_metrics"]["total_seizures"] > 0
        ]
        if per_seizure_accuracies:
            print(
                f"Mean Per-Seizure Accuracy: {np.mean(per_seizure_accuracies):.4f} (±{np.std(per_seizure_accuracies):.4f})"
            )

        fp_rates = [
            res["seizure_accuracy_metrics"]["fp_rate_per_hour"]
            for res in batch_results.values()
            if "seizure_accuracy_metrics" in res
        ]
        if fp_rates:
            print(
                f"Mean False Positive Rate: {np.mean(fp_rates):.4f} per hour (±{np.std(fp_rates):.4f})"
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

        print(f"\n✅ Batch results saved to model/batch_test_results.json")


if __name__ == "__main__":
    main()