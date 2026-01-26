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

# Import from train.py
from train import EEGDataset, CNN_LSTM_Hybrid_Dual, MetricsTracker
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
)

# Smoothing parameters
SMOOTHING_T = 10  # Window size for smoothing
SMOOTHING_X = 6   # Minimum number of positive predictions in the window


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
        Dictionary containing test metrics
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
    print(f"Initializing Dual-Stream CNN-BiLSTM model...")
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

    # Get task info from checkpoint
    checkpoint_task_mode = checkpoint.get("config", {}).get("task_mode", TASK_MODE)
    positive_class = checkpoint.get("config", {}).get(
        "positive_class", get_positive_label()
    )

    print(f"Model loaded from epoch {checkpoint.get('epoch', 'Best')}")
    print(f"Task mode: {checkpoint_task_mode.upper()} ({positive_class} vs interictal)")
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

        # Unpack 3 items: Phase, Amp, Labels
        for x_phase, x_amp, labels in pbar:
            x_phase, x_amp, labels = (
                x_phase.to(device),
                x_amp.to(device),
                labels.to(device)
            )

            # Forward pass
            outputs = model(x_phase, x_amp)
            loss = criterion(outputs, labels)

            # Track metrics
            total_loss += loss.item()
            num_batches += 1

            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            predictions = torch.argmax(outputs, dim=1)

            metrics_tracker.update(predictions, labels, probabilities)

            # Store for confusion matrix
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    metrics = metrics_tracker.compute_metrics()
    metrics["loss"] = avg_loss

    # Compute optimal threshold using Youden's J statistic
    labels_np = np.array(all_labels)
    probs_np = np.array(all_probabilities)

    # Handle edge case with only one class
    if len(np.unique(labels_np)) > 1:
        fpr, tpr, thresholds = roc_curve(labels_np, probs_np)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = 0.5

    metrics["optimal_threshold"] = float(best_threshold)

    # Calculate metrics at optimal threshold
    opt_preds = (probs_np >= best_threshold).astype(int)
    metrics["opt_accuracy"] = accuracy_score(labels_np, opt_preds)
    metrics["opt_precision"] = precision_score(labels_np, opt_preds, zero_division=0)
    metrics["opt_recall"] = recall_score(labels_np, opt_preds, zero_division=0)
    metrics["opt_f1"] = f1_score(labels_np, opt_preds, zero_division=0)

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
    )


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
        print(f"{'='*60}")

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
                # Fallback to last epoch if best not found
                last_epoch = Path(f"model/{current_output_prefix}/epoch_{TRAINING_EPOCHS:03d}.pth")
                if args.epoch is None and last_epoch.exists():
                    print(f"⚠️ 'best_model.pth' not found. Falling back to last epoch: {last_epoch}")
                    model_path = last_epoch
                else:
                    print(f"❌ Model not found: {model_path}")
                    continue
                    
            if not test_data_path.exists():
                print(f"❌ Test dataset not found: {test_data_path}")
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
            ) = evaluate_model(model_path, test_data_path, device)

            # Print results
            print("\n" + "=" * 60)
            print("PATIENT TEST RESULTS")
            print("=" * 60)
            print(f"Loss:      {metrics['loss']:.4f}")
            print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1']:.4f}")
            print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")

            print(f"\nOptimal Threshold Analysis (Youden's J):")
            print(f"Threshold: {metrics['optimal_threshold']:.4f}")
            print(f"Accuracy:  {metrics['opt_accuracy']:.4f}")
            print(f"F1 Score:  {metrics['opt_f1']:.4f}")

            print("\nConfusion Matrix:")
            print("                Predicted")
            print(f"                Interictal  {positive_class.capitalize()}")
            print(f"Actual Interictal    {cm[0,0]:6d}    {cm[0,1]:6d}")
            if cm.shape[0] > 1:
                print(f"       {positive_class.capitalize():9s}   {cm[1,0]:6d}    {cm[1,1]:6d}")

           # Save fold-specific results
            patient_results = {
                "patient_id": patient_id,
                "task_mode": checkpoint_task_mode,
                "test_metrics": metrics,
                "confusion_matrix": cm.tolist(),
                "model_path": str(model_path),
            }

            # Apply smoothing
            if SMOOTHING_T > 1:
                print(f"\nApplying smoothing with window size {SMOOTHING_T} and threshold {SMOOTHING_X}...")
                smoothed_predictions = apply_smoothing(predictions, SMOOTHING_T, SMOOTHING_X)
                smoothed_true_labels = true_labels[SMOOTHING_T - 1:]

                smoothed_metrics = {
                    "accuracy": accuracy_score(smoothed_true_labels, smoothed_predictions),
                    "precision": precision_score(smoothed_true_labels, smoothed_predictions, zero_division=0),
                    "recall": recall_score(smoothed_true_labels, smoothed_predictions, zero_division=0),
                    "f1": f1_score(smoothed_true_labels, smoothed_predictions, zero_division=0),
                }
                patient_results["smoothed_metrics"] = smoothed_metrics

                print("\n" + "=" * 60)
                print("SMOOTHED PREDICTIONS RESULTS")
                print("=" * 60)
                print(f"Smoothed_Accuracy:  {smoothed_metrics['accuracy']:.4f} ({smoothed_metrics['accuracy']*100:.2f}%)")
                print(f"Smoothed_Precision: {smoothed_metrics['precision']:.4f}")
                print(f"Smoothed_Recall:    {smoothed_metrics['recall']:.4f}")
                print(f"Smoothed_F1 Score:  {smoothed_metrics['f1']:.4f}")

            patient_results_path = Path(f"model/{current_output_prefix}/test_results.json")
            with open(patient_results_path, "w") as f:
                json.dump(patient_results, f, indent=2)

            print(f"\n✅ Results saved to {patient_results_path}")

            # Store for batch summary
            batch_results[current_idx] = {
                "patient_id": patient_id,
                "metrics": metrics,
            }
            if "smoothed_metrics" in patient_results:
                batch_results[current_idx]["smoothed_metrics"] = patient_results["smoothed_metrics"]

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

        smoothed_accuracies = [res["smoothed_metrics"]["accuracy"] for res in batch_results.values() if "smoothed_metrics" in res]
        if smoothed_accuracies:
            print(f"Mean Smoothed Accuracy: {np.mean(smoothed_accuracies):.4f} (±{np.std(smoothed_accuracies):.4f})")

        batch_summary = {
            "total_patients": n_patients,
            "patient_results": batch_results,
            "mean_accuracy": float(np.mean(accuracies)),
            "mean_auc": float(np.mean(auc_rocs)),
        }
        if smoothed_accuracies:
            batch_summary["mean_smoothed_accuracy"] = float(np.mean(smoothed_accuracies))

        with open("model/batch_test_results.json", "w") as f:
            json.dump(batch_summary, f, indent=2)

        print(f"\n✅ Batch results saved to model/batch_test_results.json")


if __name__ == "__main__":
    main()