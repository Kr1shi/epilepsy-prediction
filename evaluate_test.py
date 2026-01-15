#!/usr/bin/env python3
"""
Evaluate trained model on test dataset

Usage:
    python evaluate_test.py              # Evaluate final epoch (default)
    python evaluate_test.py --epoch 10   # Evaluate specific epoch
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from tqdm import tqdm
import json
import argparse

# Import from train.py
from train import EEGDataset, CNN_LSTM_Hybrid, MetricsTracker
from data_segmentation_helpers.config import *

def get_positive_label():
    """Get positive class label based on task mode"""
    return 'preictal' if TASK_MODE == 'prediction' else 'ictal'

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
    test_dataset = EEGDataset(test_data_path, split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=SEQUENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Initialize Deep CNN-BiLSTM model
    print(f"Initializing Deep CNN-BiLSTM model...")
    model = CNN_LSTM_Hybrid(
        num_input_channels=18,
        num_classes=2,
        sequence_length=SEQUENCE_LENGTH,
        cnn_feature_dim=512,  # Deep EEG-CNN outputs 512 features (16 conv layers)
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        lstm_num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT
    )

    # Load trained weights with compatibility for PyTorch >= 2.6 safety defaults
    print(f"Loading model checkpoint from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Get task info from checkpoint (with fallback to config)
    checkpoint_task_mode = checkpoint.get('config', {}).get('task_mode', TASK_MODE)
    positive_class = checkpoint.get('config', {}).get('positive_class', get_positive_label())

    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Task mode: {checkpoint_task_mode.upper()} ({positive_class} vs interictal)")
    print(f"Architecture: Deep CNN (16 layers, 512 features) + Bi-LSTM (3 layers, 512 hidden)")
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
        pbar = tqdm(test_loader, desc='Testing')

        for spectrograms, labels in pbar:
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            # Forward pass
            outputs = model(spectrograms)
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
    avg_loss = total_loss / num_batches
    metrics = metrics_tracker.compute_metrics()
    metrics['loss'] = avg_loss

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    return metrics, cm, all_labels, all_predictions, all_probabilities, checkpoint_task_mode, positive_class

def main():
    """Main evaluation function"""
    from data_segmentation_helpers.seizure_counts import SEIZURE_COUNTS
    from data_segmentation_helpers.config import get_lopo_config
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate trained models on test dataset')
    parser.add_argument(
        '--epoch',
        type=int,
        default=TRAINING_EPOCHS,
        help=f'Epoch number to evaluate (default: {TRAINING_EPOCHS}, the final epoch)'
    )
    parser.add_argument('--test_patient', type=str, default=None,
                        help='Patient ID to test on (None = process all patients)')
    args = parser.parse_args()
    
    # Determine which patients to process
    if args.test_patient is None:
        all_patients = sorted(SEIZURE_COUNTS.keys())
        print("="*60)
        print("BATCH EVALUATION: ALL PATIENTS (LOPO)")
        print(f"Evaluating {len(all_patients)} patients")
        print("="*60)
    else:
        all_patients = [args.test_patient]
        print("="*60)
        print("LOPO EVALUATION: SINGLE PATIENT")
        print(f"Evaluating patient {args.test_patient}")
        print("="*60)

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

    # Store results for all patients (for batch summary)
    batch_results = {}

    # Evaluate each patient
    for test_patient_id in all_patients:
        try:
            lopo_config = get_lopo_config(test_patient_id)
            current_output_prefix = lopo_config['output_prefix']

            print(f"\n{'='*60}")
            print(f"EVALUATING PATIENT {test_patient_id}")
            print(f"{'='*60}")

            # Setup patient-specific model and data
            model_path = Path(f"model/{current_output_prefix}/checkpoint_final.pth")
            dataset_dir = Path("preprocessing") / "data" / current_output_prefix
            test_data_path = dataset_dir / "test_dataset.h5"

            # Check if files exist
            if not model_path.exists():
                print(f"❌ Model not found: {model_path}")
                continue
            if not test_data_path.exists():
                print(f"❌ Test dataset not found: {test_data_path}")
                continue

            print(f"Dataset prefix: {current_output_prefix}")
            print(f"Using test dataset: {test_data_path}")

            metrics, cm, true_labels, predictions, probabilities, checkpoint_task_mode, positive_class = evaluate_model(
                model_path, test_data_path, device
            )

            # Print results
            print("\n" + "="*60)
            print("TEST RESULTS")
            print("="*60)
            print(f"Loss:      {metrics['loss']:.4f}")
            print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1']:.4f}")
            print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")

            print("\nConfusion Matrix:")
            print("                Predicted")
            print(f"                Interictal  {positive_class.capitalize()}")
            print(f"Actual Interictal    {cm[0,0]:6d}    {cm[0,1]:6d}")
            print(f"       {positive_class.capitalize():9s}   {cm[1,0]:6d}    {cm[1,1]:6d}")

            # Class-wise statistics
            print("\nClass Distribution:")
            true_np = np.array(true_labels)
            pred_np = np.array(predictions)
            print(f"True Interictal (0): {np.sum(true_np == 0)} samples")
            print(f"True {positive_class.capitalize()} (1):   {np.sum(true_np == 1)} samples")
            print(f"Pred Interictal (0): {np.sum(pred_np == 0)} samples")
            print(f"Pred {positive_class.capitalize()} (1):   {np.sum(pred_np == 1)} samples")

            # Detailed classification report
            print("\nDetailed Classification Report:")
            print(classification_report(
                true_labels,
                predictions,
                target_names=['Interictal', positive_class.capitalize()],
                digits=4
            ))

            # Save patient-specific results
            patient_results = {
                'test_patient': test_patient_id,
                'task_mode': checkpoint_task_mode,
                'positive_class': positive_class,
                'negative_class': 'interictal',
                'test_metrics': metrics,
                'confusion_matrix': cm.tolist(),
                'model_path': str(model_path),
                'test_data_path': str(test_data_path)
            }

            patient_results_path = Path(f"model/{current_output_prefix}/test_results.json")
            patient_results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(patient_results_path, 'w') as f:
                json.dump(patient_results, f, indent=2)

            print(f"\n✅ Results saved to {patient_results_path}")

            # Store for batch summary
            batch_results[test_patient_id] = {
                'test_patient': test_patient_id,
                'output_prefix': current_output_prefix,
                'metrics': metrics,
                'confusion_matrix': cm.tolist()
            }

        except Exception as e:
            print(f"❌ Error evaluating patient {test_patient_id}: {e}")
            import traceback
            traceback.print_exc()

    # Save batch results summary if processing multiple patients
    if args.test_patient is None and batch_results:
        print("\n" + "="*60)
        print("BATCH EVALUATION SUMMARY (LOPO)")
        print("="*60)

        # Compute aggregate metrics
        accuracies = [res['metrics']['accuracy'] for res in batch_results.values()]
        precisions = [res['metrics']['precision'] for res in batch_results.values()]
        recalls = [res['metrics']['recall'] for res in batch_results.values()]
        f1_scores = [res['metrics']['f1'] for res in batch_results.values()]
        auc_rocs = [res['metrics']['auc_roc'] for res in batch_results.values()]

        print(f"Patients evaluated: {len(batch_results)}/{len(all_patients)}")
        print(f"\nMean metrics (across patients):")
        print(f"  Accuracy:  {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
        print(f"  Precision: {np.mean(precisions):.4f} (±{np.std(precisions):.4f})")
        print(f"  Recall:    {np.mean(recalls):.4f} (±{np.std(recalls):.4f})")
        print(f"  F1 Score:  {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
        print(f"  AUC-ROC:   {np.mean(auc_rocs):.4f} (±{np.std(auc_rocs):.4f})")

        # Save batch summary
        batch_summary = {
            'total_patients': len(all_patients),
            'evaluated_patients': len(batch_results),
            'patient_results': batch_results,
            'aggregate_metrics': {
                'accuracy': {
                    'mean': float(np.mean(accuracies)),
                    'std': float(np.std(accuracies))
                },
                'precision': {
                    'mean': float(np.mean(precisions)),
                    'std': float(np.std(precisions))
                },
                'recall': {
                    'mean': float(np.mean(recalls)),
                    'std': float(np.std(recalls))
                },
                'f1_score': {
                    'mean': float(np.mean(f1_scores)),
                    'std': float(np.std(f1_scores))
                },
                'auc_roc': {
                    'mean': float(np.mean(auc_rocs)),
                    'std': float(np.std(auc_rocs))
                }
            }
        }

        batch_results_path = Path("model/lopo_batch_test_results.json")
        with open(batch_results_path, 'w') as f:
            json.dump(batch_summary, f, indent=2)

        print(f"\n✅ Batch results saved to {batch_results_path}")
        print("="*60)

if __name__ == "__main__":
    main()
