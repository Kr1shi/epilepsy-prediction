#!/usr/bin/env python3
"""
Evaluate trained model on test dataset

Usage:
    python evaluate_test.py              # Evaluate best model (default)
    python evaluate_test.py --epoch 10   # Evaluate specific epoch
"""
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from tqdm import tqdm
import json
import argparse
import matplotlib.pyplot as plt

# Import from train.py
from train import EEGDataset, CNN_LSTM_Hybrid_Dual, MetricsTracker
# from total_accuracy import calculate_per_seizure_accuracy
from data_segmentation_helpers.config import (
    SEQUENCE_LENGTH,
    SEQUENCE_BATCH_SIZE,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
    TRAINING_EPOCHS,
    PATIENTS,
    PATIENT_INDEX,
    get_patient_config,
    MAX_HORIZON_SEC,
)

def evaluate_model(model_path, test_data_path, device):
    """
    Load trained model and evaluate TTS regression on test dataset
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

    # Initialize Dual-Stream Model (Regression)
    model = CNN_LSTM_Hybrid_Dual(
        num_input_channels=18,
        num_classes=1, # Regression
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

    # Evaluate
    print("\nEvaluating TTS Regression on test set...")
    metrics_tracker = MetricsTracker()
    
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")

        for x_phase, x_amp, labels in pbar:
            x_phase, x_amp, labels = (
                x_phase.to(device),
                x_amp.to(device),
                labels.to(device).unsqueeze(1),
            )

            outputs = model(x_phase, x_amp)
            metrics_tracker.update(outputs, labels)

            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # Compute metrics
    metrics = metrics_tracker.compute_metrics()

    return metrics, all_labels, all_predictions


def plot_tts_countdown(true_tts, pred_tts, patient_id, output_dir):
    """
    Plots Predicted TTS vs Ground Truth Sawtooth.
    Saves to 'result_plots/{patient_id}/tts_countdown.png'
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert seconds to minutes for plotting
    true_mins = np.array(true_tts) / 60.0
    pred_mins = np.array(pred_tts) / 60.0
    
    plt.figure(figsize=(20, 8))
    
    # Plot entire test set sequence
    x_axis = np.arange(len(true_mins))
    
    plt.plot(x_axis, true_mins, label="Ground Truth (Sawtooth)", color="black", alpha=0.5, linestyle="--")
    plt.plot(x_axis, pred_mins, label="Model Prediction", color="red", linewidth=1, alpha=0.8)
    
    # Alarm threshold at 15 minutes
    plt.axhline(y=15, color="blue", linestyle=":", label="Alarm Threshold (15m)")
    
    plt.title(f"TTS Regression Countdown: Patient {patient_id}", fontsize=16)
    plt.xlabel("Sample Index (Shuffled Test Set)", fontsize=12)
    plt.ylabel("Minutes to Seizure", fontsize=12)
    plt.ylim(-5, (MAX_HORIZON_SEC/60) + 10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = output_dir / "tts_countdown_plot.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved TTS countdown plot to {save_path}")


def main():
    """Main TTS Regression evaluation function"""
    parser = argparse.ArgumentParser(
        description="Evaluate TTS Regression model on test dataset"
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
    else:
        patients_to_process = [PATIENT_INDEX]

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using Device: {device}")

    batch_results = {}

    for current_idx in patients_to_process:
        patient_config = get_patient_config(current_idx)
        patient_id = patient_config["patient_id"]
        output_prefix = patient_config["output_prefix"]

        print(f"\n{'='*60}\nEVALUATING PATIENT: {patient_id}\n{'='*60}")

        try:
            # Paths
            model_filename = f"epoch_{args.epoch:03d}.pth" if args.epoch else "best_model.pth"
            model_path = Path(f"model/tts_regression/{output_prefix}/{model_filename}")
            test_data_path = Path(f"preprocessing/data/{output_prefix}/s{patient_id}_dataset.h5")

            if not model_path.exists():
                print(f"❌ Model not found: {model_path}")
                continue
            if not test_data_path.exists():
                print(f"❌ Test dataset not found: {test_data_path}")
                continue

            # Run Evaluation
            metrics, true_labels, predictions = evaluate_model(model_path, test_data_path, device)

            print("\n" + "=" * 60)
            print("TEST RESULTS (MINUTES)")
            print("=" * 60)
            print(f"MAE:  {metrics['mae_minutes']:.2f} min")
            print(f"RMSE: {metrics['rmse_minutes']:.2f} min")

            # Plotting
            output_dir = Path(f"result_plots/tts_regression/{patient_id}")
            plot_tts_countdown(true_labels, predictions, patient_id, output_dir)

            # Save results
            batch_results[patient_id] = metrics
            res_path = Path(f"model/tts_regression/{output_prefix}/test_results.json")
            with open(res_path, "w") as f:
                json.dump(metrics, f, indent=2)

        except Exception as e:
            print(f"❌ Error evaluating {patient_id}: {e}")
            import traceback
            traceback.print_exc()

    # Final Summary
    if len(batch_results) > 1:
        print("\n" + "=" * 60)
        print("BATCH EVALUATION SUMMARY")
        print("=" * 60)
        avg_mae = np.mean([r['mae_minutes'] for r in batch_results.values()])
        avg_rmse = np.mean([r['rmse_minutes'] for r in batch_results.values()])
        print(f"Mean MAE Across Patients:  {avg_mae:.2f} min")
        print(f"Mean RMSE Across Patients: {avg_rmse:.2f} min")

if __name__ == "__main__":
    main()
