#!/usr/bin/env python3
"""
Leave-One-Seizure-Out Cross-Validation (LOSO-CV)

For each patient with K seizures, trains K models — each holding out one seizure
for test. Reports per-fold and mean metrics across all folds.

Prerequisites:
    python data_segmentation.py
    python data_preprocessing.py --master

Usage:
    python loso_cv.py                    # All patients
    python loso_cv.py --patient chb01    # Single patient
"""
import argparse
import json
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_segmentation_helpers.config import (
    CONV_EMBEDDING_DIM,
    FINETUNING_LEARNING_RATE,
    INTERICTAL_TO_PREICTAL_RATIO,
    PATIENTS,
    PATIENT_INDEX,
    SEGMENT_DURATION,
    SEQUENCE_BATCH_SIZE,
    SEQUENCE_LENGTH,
    TASK_MODE,
    TRAINING_EPOCHS,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_FFN_DIM,
    TRANSFORMER_NUM_HEADS,
    TRANSFORMER_NUM_LAYERS,
    USE_CLS_TOKEN,
    WEIGHT_DECAY,
    get_patient_config,
)
from train import ConvTransformerModel, set_seed


# =============================================================================
# FoldDataset: reads from master HDF5 with on-the-fly normalization
# =============================================================================


class FoldDataset(Dataset):
    """Dataset that reads specific indices from a master unnormalized HDF5."""

    def __init__(self, master_h5_path, indices, mean, std, augment=False):
        self.h5_path = str(master_h5_path)
        self.indices = sorted(indices)  # HDF5 fancy indexing needs sorted order
        # Map from position in self.indices to original HDF5 index
        self.mean = mean
        self.std = std
        self.augment = augment
        self.h5_file = None

        with h5py.File(self.h5_path, "r") as f:
            self.labels = torch.LongTensor(f["labels"][self.indices])

        self.length = len(self.indices)

    def _open_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.length

    def _apply_augmentation(self, x):
        """SpecAugment-style augmentation."""
        seq_len = x.shape[0]
        freq_bins = x.shape[2]

        for _ in range(torch.randint(1, 3, (1,)).item()):
            if torch.rand(1).item() < 0.5:
                mask_len = torch.randint(1, max(2, seq_len // 8), (1,)).item()
                start = torch.randint(0, seq_len - mask_len + 1, (1,)).item()
                x[start : start + mask_len] = 0.0

        for _ in range(torch.randint(1, 3, (1,)).item()):
            if torch.rand(1).item() < 0.5:
                mask_len = torch.randint(1, max(2, freq_bins // 6), (1,)).item()
                start = torch.randint(0, freq_bins - mask_len + 1, (1,)).item()
                x[:, :, start : start + mask_len, :] = 0.0

        if torch.rand(1).item() < 0.3:
            n_channels = x.shape[1]
            n_mask = torch.randint(1, min(4, n_channels), (1,)).item()
            channels_to_mask = torch.randperm(n_channels)[:n_mask]
            x[:, channels_to_mask, :, :] = 0.0

        if torch.rand(1).item() < 0.5:
            noise_std = 0.1 * x.std()
            x = x + torch.randn_like(x) * noise_std

        return x

    def __getitem__(self, idx):
        self._open_h5()
        h5_idx = self.indices[idx]
        x = torch.FloatTensor(self.h5_file["spectrograms"][h5_idx])
        # On-the-fly normalization
        x = (x - self.mean) / self.std if self.std > 1e-8 else x - self.mean
        if self.augment:
            x = self._apply_augmentation(x)
        return x, self.labels[idx]

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()


# =============================================================================
# Fold splitting logic
# =============================================================================


def compute_fold_stats(master_h5_path, train_indices):
    """Compute mean/std from training indices for z-score normalization."""
    if not train_indices:
        return 0.0, 1.0  # Safe defaults for empty train set
    sorted_idx = sorted(train_indices)
    with h5py.File(master_h5_path, "r") as f:
        sum_v, sum_sq, count = 0.0, 0.0, 0
        chunk_size = 50
        for i in range(0, len(sorted_idx), chunk_size):
            batch_idx = sorted_idx[i : i + chunk_size]
            chunk = f["spectrograms"][batch_idx].astype(np.float64)
            sum_v += np.sum(chunk)
            sum_sq += np.sum(chunk**2)
            count += chunk.size
    mean = sum_v / count
    std = np.sqrt(max((sum_sq / count) - (mean**2), 0.0))
    return float(mean), max(float(std), 1e-8)


def get_loso_fold_splits(master_h5_path, test_seizure_id, random_seed=42):
    """Split master HDF5 indices into train/val/test for a given test seizure.

    Test: all sequences from the held-out seizure (positive + associated interictal).
    Remaining sequences are mixed and split 90/10 at the sequence level for train/val.
    Train is class-balanced; val is kept unbalanced (needs both classes for AUC).

    Returns dict with keys 'train', 'val', 'test', each a list of HDF5 indices.
    """
    rng = random.Random(random_seed)

    with h5py.File(master_h5_path, "r") as f:
        labels = f["labels"][:]
        seizure_ids = f["segment_info/seizure_ids"][:]
        global_starts = f["segment_info/sequence_start_global"][:]

    n = len(labels)
    all_seizure_ids = sorted(set(int(s) for s in seizure_ids if s >= 0))

    # --- Test split: all sequences from the test seizure + nearby interictal ---
    test_positive = [i for i in range(n) if labels[i] == 1 and seizure_ids[i] == test_seizure_id]

    # Associate interictal with nearest seizure (for test split only)
    seizure_anchors = {}
    for i in range(n):
        if labels[i] == 1:
            sid = int(seizure_ids[i])
            t = global_starts[i]
            if sid not in seizure_anchors or t < seizure_anchors[sid]:
                seizure_anchors[sid] = t

    sorted_anchors = sorted(
        [(sid, seizure_anchors[sid]) for sid in all_seizure_ids if sid in seizure_anchors],
        key=lambda x: x[1],
    )

    test_interictal = []
    non_test_indices = []
    for i in range(n):
        if labels[i] == 1 and seizure_ids[i] == test_seizure_id:
            continue  # already in test_positive
        if labels[i] == 0:
            # Assign interictal to nearest following seizure
            t = global_starts[i]
            assigned_sid = None
            for sid, anchor_t in sorted_anchors:
                if anchor_t > t:
                    assigned_sid = sid
                    break
            if assigned_sid is None:
                assigned_sid = min(sorted_anchors, key=lambda x: abs(x[1] - t))[0]

            if assigned_sid == test_seizure_id:
                test_interictal.append(i)
                continue
        non_test_indices.append(i)

    # --- Train/Val split: mix remaining sequences, split 90/10 at sequence level ---
    # Separate by class for stratified split
    remaining_positive = [i for i in non_test_indices if labels[i] == 1]
    remaining_interictal = [i for i in non_test_indices if labels[i] == 0]

    rng.shuffle(remaining_positive)
    rng.shuffle(remaining_interictal)

    # 10% of each class goes to val (stratified)
    n_val_pos = max(1, round(len(remaining_positive) * 0.1))
    n_val_inter = max(1, round(len(remaining_interictal) * 0.1))

    val_positive = remaining_positive[:n_val_pos]
    val_interictal = remaining_interictal[:n_val_inter]
    train_positive = remaining_positive[n_val_pos:]
    train_interictal = remaining_interictal[n_val_inter:]

    # Balance train (downsample majority class)
    def balance(pos_idx, inter_idx):
        if not pos_idx or not inter_idx:
            return pos_idx + inter_idx
        rng_bal = random.Random(random_seed + len(pos_idx))
        rng_bal.shuffle(inter_idx)
        if len(inter_idx) >= len(pos_idx):
            target = min(int(len(pos_idx) * INTERICTAL_TO_PREICTAL_RATIO), len(inter_idx))
            return pos_idx + inter_idx[:target]
        else:
            rng_bal.shuffle(pos_idx)
            target = min(int(len(inter_idx) / INTERICTAL_TO_PREICTAL_RATIO), len(pos_idx))
            return pos_idx[:target] + inter_idx

    train_balanced = balance(train_positive, train_interictal)
    val_all = val_positive + val_interictal

    return {
        "train": train_balanced,
        "val": val_all,
        "test": test_positive + test_interictal,
    }


# =============================================================================
# Per-fold training and evaluation
# =============================================================================


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_fold(
    master_h5_path,
    patient_id,
    fold_seizure_id,
    pretrained_path,
    device,
    random_seed=42,
    no_finetune=False,
):
    """Train a model for one LOSO fold. Returns test metrics dict."""
    set_seed(random_seed)

    # 1. Split indices
    splits = get_loso_fold_splits(master_h5_path, fold_seizure_id, random_seed)
    n_train, n_val, n_test = len(splits["train"]), len(splits["val"]), len(splits["test"])
    print(f"  Fold seizure {fold_seizure_id}: train={n_train}, val={n_val}, test={n_test}", flush=True)

    if n_test == 0:
        print(f"  WARNING: No test samples for seizure {fold_seizure_id}, skipping")
        return None

    if n_train == 0:
        print(f"  WARNING: No training samples for seizure {fold_seizure_id}, skipping")
        return None

    # 2. Compute normalization stats from training set
    mean, std = compute_fold_stats(master_h5_path, splits["train"])

    # 3. Create datasets
    val_ds = FoldDataset(master_h5_path, splits["val"], mean, std, augment=False)
    test_ds = FoldDataset(master_h5_path, splits["test"], mean, std, augment=False)

    val_loader = DataLoader(val_ds, batch_size=SEQUENCE_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=SEQUENCE_BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. Initialize model with pretrained weights
    model = ConvTransformerModel(
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

    if pretrained_path and Path(pretrained_path).exists():
        model.load_state_dict(
            torch.load(pretrained_path, map_location=device, weights_only=False)
        )

    model.to(device)
    model.eval()

    # 5. Compute val predictions (for threshold tuning)
    val_labels_np = np.array([])
    val_probs_np = np.array([])

    val_probs, val_labels = [], []
    with torch.no_grad():
        for x, labels in val_loader:
            x, labels = x.to(device), labels.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            val_probs.extend(probs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    val_labels_np = np.array(val_labels)
    val_probs_np = np.array(val_probs)

    if not no_finetune:
        # Freeze conv tower during fine-tuning
        for param in model.conv_tower.parameters():
            param.requires_grad = False

        # Loss, optimizer
        train_ds = FoldDataset(master_h5_path, splits["train"], mean, std, augment=True)
        train_loader = DataLoader(train_ds, batch_size=SEQUENCE_BATCH_SIZE, shuffle=True, num_workers=0)

        train_labels = train_ds.labels
        counts = torch.bincount(train_labels).float()
        if len(counts) < 2:
            counts = torch.tensor([1.0, 1.0])
        class_weights = counts.sum() / (len(counts) * counts)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device), label_smoothing=0.05
        )

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=FINETUNING_LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
        )

        # Training loop
        best_val_auc = 0.0
        best_state = None
        epochs_no_improve = 0

        for epoch in range(TRAINING_EPOCHS):
            model.train()
            total_loss = 0.0
            for x, labels in train_loader:
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            if np.isnan(total_loss):
                print(f"    WARNING: NaN loss at epoch {epoch}, stopping training", flush=True)
                break

            model.eval()
            val_probs, val_labels = [], []
            with torch.no_grad():
                for x, labels in val_loader:
                    x, labels = x.to(device), labels.to(device)
                    outputs = model(x)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    val_probs.extend(probs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            val_labels_np = np.array(val_labels)
            val_probs_np = np.array(val_probs)

            if np.any(np.isnan(val_probs_np)):
                print(f"    WARNING: NaN in val probs at epoch {epoch}, stopping training", flush=True)
                break

            if len(np.unique(val_labels_np)) > 1:
                val_auc = roc_auc_score(val_labels_np, val_probs_np)
            else:
                val_auc = 0.5

            scheduler.step(val_auc)

            avg_loss = total_loss / max(len(train_loader), 1)
            marker = ""
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                marker = " *"
            else:
                epochs_no_improve += 1

            print(f"    Ep {epoch+1:>2}/{TRAINING_EPOCHS}  loss={avg_loss:.4f}  val_auc={val_auc:.4f}  best={best_val_auc:.4f}{marker}", flush=True)

            if epochs_no_improve >= 5:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(device)
        model.eval()

    test_probs, test_labels = [], []
    with torch.no_grad():
        for x, labels in test_loader:
            x, labels = x.to(device), labels.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_labels_np = np.array(test_labels)
    test_probs_np = np.array(test_probs)

    # Guard against NaN in test probabilities
    if np.any(np.isnan(test_probs_np)):
        print(f"    WARNING: NaN in test probs, replacing with 0.5")
        test_probs_np = np.nan_to_num(test_probs_np, nan=0.5)

    # Compute threshold on val set
    if len(np.unique(val_labels_np)) > 1 and not np.any(np.isnan(val_probs_np)):
        fpr, tpr, thresholds = roc_curve(val_labels_np, val_probs_np)
        j_scores = tpr - fpr
        threshold = float(thresholds[np.argmax(j_scores)])
    else:
        threshold = 0.5

    test_preds = (test_probs_np >= threshold).astype(int)

    # Compute test metrics
    if len(np.unique(test_labels_np)) > 1:
        test_auc = roc_auc_score(test_labels_np, test_probs_np)
    else:
        test_auc = 0.5

    metrics = {
        "seizure_id": int(fold_seizure_id),
        "auc_roc": float(test_auc),
        "accuracy": float(accuracy_score(test_labels_np, test_preds)),
        "precision": float(precision_score(test_labels_np, test_preds, zero_division=0)),
        "recall": float(recall_score(test_labels_np, test_preds, zero_division=0)),
        "f1": float(f1_score(test_labels_np, test_preds, zero_division=0)),
        "threshold": threshold,
        "best_val_auc": float(best_val_auc) if not no_finetune else -1,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_test_positive": int(np.sum(test_labels_np == 1)),
        "n_test_negative": int(np.sum(test_labels_np == 0)),
    }

    # Per-seizure detection: did ANY preictal sample get predicted correctly?
    preictal_mask = test_labels_np == 1
    if preictal_mask.any():
        seizure_detected = int(np.any(test_preds[preictal_mask] == 1))
        metrics["seizure_detected"] = seizure_detected
    else:
        metrics["seizure_detected"] = -1  # No preictal in test

    print(f"    AUC={metrics['auc_roc']:.4f}  Acc={metrics['accuracy']:.4f}  "
          f"F1={metrics['f1']:.4f}  Detected={metrics['seizure_detected']}", flush=True)

    return metrics


# =============================================================================
# Main: run LOSO-CV for all patients
# =============================================================================


def run_patient_loso_cv(patient_id, pretrained_path, device, no_finetune=False):
    """Run full LOSO-CV for one patient. Returns list of per-fold metrics."""
    master_h5 = Path("preprocessing") / "data" / patient_id / "master_unnormalized.h5"
    if not master_h5.exists():
        print(f"  ERROR: {master_h5} not found. Run: python data_preprocessing.py --master")
        return None

    # Get seizure IDs from master HDF5
    with h5py.File(master_h5, "r") as f:
        seizure_ids_arr = f["segment_info/seizure_ids"][:]
    unique_seizures = sorted(set(int(s) for s in seizure_ids_arr if s >= 0))

    if len(unique_seizures) < 2:
        print(f"  WARNING: {patient_id} has {len(unique_seizures)} seizure(s), skipping LOSO-CV")
        return None

    print(f"\n{'='*60}", flush=True)
    print(f"LOSO-CV: {patient_id} — {len(unique_seizures)} seizures: {unique_seizures}", flush=True)
    print(f"{'='*60}", flush=True)

    fold_results = []
    for fold_idx, test_seizure_id in enumerate(unique_seizures):
        print(f"\n  --- Fold {fold_idx+1}/{len(unique_seizures)} (test seizure {test_seizure_id}) ---")
        metrics = train_fold(
            master_h5_path=str(master_h5),
            patient_id=patient_id,
            fold_seizure_id=test_seizure_id,
            pretrained_path=pretrained_path,
            device=device,
            random_seed=42 + fold_idx,
            no_finetune=no_finetune,
        )
        if metrics is not None:
            fold_results.append(metrics)

    return fold_results


def main():
    parser = argparse.ArgumentParser(description="LOSO Cross-Validation")
    parser.add_argument("--patient", type=str, default=None,
                        help="Run for a single patient (e.g., chb01)")
    parser.add_argument("--pretrained", type=str, default="model/pretrained_encoder.pth",
                        help="Path to pretrained encoder weights")
    parser.add_argument("--no-finetune", action="store_true",
                        help="Skip fine-tuning, evaluate pretrained model directly")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Pretrained encoder: {args.pretrained}")
    if args.no_finetune:
        print("Mode: pretrained-only (no fine-tuning)")
    start_time = time.time()

    if args.patient:
        patients = [args.patient]
    elif PATIENT_INDEX is not None:
        patients = [PATIENTS[PATIENT_INDEX]]
    else:
        patients = PATIENTS

    all_results = {}
    all_fold_aucs = []

    for patient_id in patients:
        fold_results = run_patient_loso_cv(patient_id, args.pretrained, device, args.no_finetune)
        if fold_results is None or len(fold_results) == 0:
            continue

        # Per-patient summary
        aucs = [r["auc_roc"] for r in fold_results]
        accs = [r["accuracy"] for r in fold_results]
        f1s = [r["f1"] for r in fold_results]
        detected = [r["seizure_detected"] for r in fold_results if r["seizure_detected"] >= 0]

        patient_summary = {
            "patient_id": patient_id,
            "n_folds": len(fold_results),
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "mean_accuracy": float(np.mean(accs)),
            "mean_f1": float(np.mean(f1s)),
            "seizure_detection_rate": float(np.mean(detected)) if detected else 0.0,
            "per_fold": fold_results,
        }
        all_results[patient_id] = patient_summary
        all_fold_aucs.extend(aucs)

        print(f"\n  {patient_id} LOSO-CV Summary:")
        print(f"    Mean AUC: {patient_summary['mean_auc']:.4f} (+/- {patient_summary['std_auc']:.4f})")
        print(f"    Mean Acc: {patient_summary['mean_accuracy']:.4f}")
        print(f"    Mean F1:  {patient_summary['mean_f1']:.4f}")
        print(f"    Seizure Detection: {patient_summary['seizure_detection_rate']:.2%}")

    # Global summary
    print(f"\n{'='*60}")
    print(f"OVERALL LOSO-CV SUMMARY ({len(all_results)} patients)")
    print(f"{'='*60}")

    if all_fold_aucs:
        patient_mean_aucs = [r["mean_auc"] for r in all_results.values()]
        print(f"Mean AUC (across patients): {np.mean(patient_mean_aucs):.4f} (+/- {np.std(patient_mean_aucs):.4f})")
        print(f"Mean AUC (across all folds): {np.mean(all_fold_aucs):.4f} (+/- {np.std(all_fold_aucs):.4f})")
        print(f"Total folds evaluated: {len(all_fold_aucs)}")

        detection_rates = [r["seizure_detection_rate"] for r in all_results.values()]
        print(f"Mean Seizure Detection Rate: {np.mean(detection_rates):.2%}")

        # Per-patient breakdown
        print(f"\n{'Patient':<10} {'Folds':>5} {'Mean AUC':>10} {'Std AUC':>10} {'Detection':>10}")
        print("-" * 50)
        for pid, res in sorted(all_results.items()):
            print(f"{pid:<10} {res['n_folds']:>5} {res['mean_auc']:>10.4f} {res['std_auc']:>10.4f} "
                  f"{res['seizure_detection_rate']:>9.1%}")

    # Save results
    output_dir = Path("model") / "loso_cv"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "loso_cv_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "pretrained_path": args.pretrained,
                "total_patients": len(all_results),
                "total_folds": len(all_fold_aucs),
                "mean_auc_across_patients": float(np.mean(patient_mean_aucs)) if all_fold_aucs else 0,
                "mean_auc_across_folds": float(np.mean(all_fold_aucs)) if all_fold_aucs else 0,
                "patients": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_path}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} min")


if __name__ == "__main__":
    main()
