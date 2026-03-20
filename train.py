import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import h5py
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm
import time
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

import gc

from data_segmentation_helpers.config import *
from data_segmentation_helpers.config import PRETRAIN_ONLY_PATIENTS


class FocalLoss(nn.Module):
    """Focal Loss: down-weights easy examples, focuses on hard ones."""

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        n_classes = inputs.size(1)
        smooth_targets = torch.full_like(inputs, self.label_smoothing / (n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        focal_weight = (1.0 - probs) ** self.gamma
        loss = -focal_weight * log_probs * smooth_targets

        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)

        return loss.sum(dim=1).mean()


class EEGDataset(Dataset):
    """Single-Stream Dataset with lazy HDF5 loading.

    Labels in HDF5: 0=interictal, 1=preictal, 2=ictal.
    Stage filtering remaps to binary for training:
      - stage="pretrain": keeps ictal(2)+interictal(0), remaps 2→1
      - stage="finetune": keeps preictal(1)+interictal(0), labels already 0/1
      - stage=None: keeps all (no filtering)
    """

    def __init__(self, h5_file_path, split="train", augment=False, stage=None):
        self.h5_file_path = h5_file_path
        self.split = split
        self.h5_file = None
        self.augment = augment and (split == "train")
        self.mean = None
        self.std = None

        with h5py.File(h5_file_path, "r") as f:
            raw_labels = torch.LongTensor(f["labels"][:])

            if "metadata" in f:
                self.metadata = dict(f["metadata"].attrs)
                self.stats = {
                    "sum_v": float(f["metadata"].attrs.get("sum_v", 0.0)),
                    "sum_sq": float(f["metadata"].attrs.get("sum_sq", 0.0)),
                    "count": float(f["metadata"].attrs.get("count", 0.0)),
                }
            else:
                self.metadata = {}
                self.stats = {"sum_v": 0.0, "sum_sq": 0.0, "count": 0.0}

        # Stage-based filtering
        if stage == "pretrain":
            # Keep ictal (2) + interictal (0), remap ictal 2→1
            mask = (raw_labels == 0) | (raw_labels == 2)
            self.valid_indices = torch.where(mask)[0]
            self.labels = (raw_labels[self.valid_indices] == 2).long()
        elif stage == "finetune":
            # Keep preictal (1) + interictal (0)
            mask = (raw_labels == 0) | (raw_labels == 1)
            self.valid_indices = torch.where(mask)[0]
            self.labels = raw_labels[self.valid_indices]
        else:
            self.valid_indices = torch.arange(len(raw_labels))
            self.labels = raw_labels

        self.length = len(self.labels)

        print(f"Loaded {self.split} dataset: {self.length} samples (stage={stage})")
        print(f"  - Class Dist: {torch.bincount(self.labels)}")
        if self.augment:
            print(f"  - Augmentation: ENABLED")

    def set_normalization(self, mean, std):
        """Set normalization parameters (computed from training data only)."""
        self.mean = mean
        self.std = std

    def _open_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_file_path, "r")

    def __len__(self):
        return self.length

    def _apply_augmentation(self, x):
        """Apply spectrogram augmentation.
        x: (sequence_length, channels, freq_bins, time_bins)
        """
        # Time masking: zero out random contiguous segments in the sequence
        if torch.rand(1).item() < 0.5:
            seq_len = x.shape[0]
            mask_len = torch.randint(1, max(2, seq_len // 10), (1,)).item()
            start = torch.randint(0, seq_len - mask_len + 1, (1,)).item()
            x[start : start + mask_len] = 0.0

        # Frequency masking: zero out random frequency bands
        if torch.rand(1).item() < 0.5:
            freq_bins = x.shape[2]
            mask_len = torch.randint(1, max(2, freq_bins // 8), (1,)).item()
            start = torch.randint(0, freq_bins - mask_len + 1, (1,)).item()
            x[:, :, start : start + mask_len, :] = 0.0

        # Gaussian noise
        if torch.rand(1).item() < 0.5:
            noise_std = 0.05 * x.std()
            x = x + torch.randn_like(x) * noise_std

        return x

    def __getitem__(self, idx):
        """Returns: (spectrogram, label)
        spectrogram: (sequence_length, channels, freq_bins, time_bins)
        """
        self._open_h5()
        real_idx = self.valid_indices[idx].item()
        x = torch.FloatTensor(self.h5_file["spectrograms"][real_idx])
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std if self.std > 1e-8 else x - self.mean
        if self.augment:
            x = self._apply_augmentation(x)
        return x, self.labels[idx]

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()


class ConvTower(nn.Module):
    """Conv tower that extracts a fixed-size embedding from a single spectrogram segment.

    Input:  (batch, 18, 50, 50)  — channels, freq_bins, time_bins per 5s segment
    Output: (batch, embed_dim)

    Feature map progression: (50,50) → (25,25) → (13,13) → AdaptiveAvgPool → (1,1)
    Uses strided convolutions instead of MaxPool to learn downsampling.
    """

    def __init__(self, in_channels=18, embed_dim=128):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class ConvGRUModel(nn.Module):
    """Single-stream Conv-BiGRU for EEG sequence classification.

    Input: (batch, 30, 18, 50, 50) — 30 segments × 5s = 2.5 min sequences.
    Conv tower extracts a 128-dim embedding from each 5s spectrogram segment.
    BiGRU processes the sequence of embeddings to capture the temporal
    evolution of preictal EEG (gradual buildup toward seizure onset).
    Final hidden states from both directions are concatenated and classified.
    """

    def __init__(
        self,
        num_input_channels=18,
        num_classes=2,
        embed_dim=CONV_EMBEDDING_DIM,
        gru_hidden=GRU_HIDDEN_DIM,
        gru_layers=GRU_NUM_LAYERS,
        dropout=GRU_DROPOUT,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.gru_hidden = gru_hidden

        # Per-segment feature extractor
        self.conv_tower = ConvTower(in_channels=num_input_channels, embed_dim=embed_dim)

        # BiLSTM: processes sequence of segment embeddings
        self.gru = nn.LSTM(
            input_size=embed_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0,
        )

        # Layer norm on LSTM output
        self.norm = nn.LayerNorm(gru_hidden * 2)

        # Classification head (input = 2 * gru_hidden for bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, channels, freq_bins, time_bins)
        Returns:
            logits: (batch, num_classes)
        """
        B, S, C, H, W = x.shape

        # Flatten sequence dimension for CNN processing
        x_flat = x.view(B * S, C, H, W)

        # Conv tower: (B*S, embed_dim)
        embeddings = self.conv_tower(x_flat)

        # Reshape to sequence: (B, S, embed_dim)
        embeddings = embeddings.view(B, S, self.embed_dim)

        # BiLSTM: output shape (B, S, 2*gru_hidden), (h_n, c_n) where h_n is (2*layers, B, gru_hidden)
        _, (h_n, _) = self.gru(embeddings)

        # Concatenate final hidden states from forward and backward directions
        # h_n shape: (2*num_layers, B, gru_hidden) → take last layer's forward and backward
        h_forward = h_n[-2]  # (B, gru_hidden)
        h_backward = h_n[-1]  # (B, gru_hidden)
        pooled = torch.cat([h_forward, h_backward], dim=1)  # (B, 2*gru_hidden)

        pooled = self.norm(pooled)

        return self.fc(pooled)


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.true_labels = []
        self.probabilities = []

    def update(self, predictions, labels, probabilities):
        self.predictions.extend(predictions.detach().cpu().numpy())
        self.true_labels.extend(labels.detach().cpu().numpy())
        self.probabilities.extend(probabilities.detach().cpu().numpy())

    def compute_metrics(self):
        pred_np = np.array(self.predictions)
        true_np = np.array(self.true_labels)
        prob_np = np.array(self.probabilities)

        if len(np.unique(true_np)) > 1:
            auc = roc_auc_score(true_np, prob_np)
            fpr, tpr, thresholds = roc_curve(true_np, prob_np)
            j_scores = tpr - fpr
            best_threshold = thresholds[np.argmax(j_scores)]
        else:
            auc = 0.5
            best_threshold = 0.5

        return {
            "accuracy": accuracy_score(true_np, pred_np),
            "precision": precision_score(
                true_np, pred_np, average="binary", zero_division=0
            ),
            "recall": recall_score(true_np, pred_np, average="binary", zero_division=0),
            "f1": f1_score(true_np, pred_np, average="binary", zero_division=0),
            "auc_roc": auc,
            "optimal_threshold": float(best_threshold),
        }


def optimize_running_average(predictions, labels, max_window=30):
    """Finds the optimal Window Size and Count Threshold for smoothing hard predictions."""
    best_f1 = 0
    best_params = (1, 1)

    preds_np = np.array(predictions)
    labels_np = np.array(labels)

    # Ensure same length (safety check for edge cases with very small datasets)
    min_len = min(len(preds_np), len(labels_np))
    preds_np = preds_np[:min_len]
    labels_np = labels_np[:min_len]

    if min_len < 10:
        print(f"  Skipping smoothing optimization (only {min_len} samples)")
        return best_params

    for window_size in range(5, min(max_window, min_len) + 1):
        kernel = np.ones(window_size)
        smoothed_sums = np.convolve(preds_np, kernel, mode="same")

        for count_threshold in range(1, window_size + 1):
            final_preds = (smoothed_sums >= count_threshold).astype(int)
            current_f1 = f1_score(labels_np, final_preds, zero_division=0)

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_params = (window_size, count_threshold)

    print(
        f"  Best Smoothing Params (Train): Window={best_params[0]}, Threshold={best_params[1]}, F1={best_f1:.4f}"
    )
    return best_params

def compute_normalization_stats(datasets):
    """Combine per-split stats from multiple datasets to compute mean/std.

    Only pass training datasets to this function to avoid data leakage.
    """
    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0.0
    for ds in datasets:
        total_sum += ds.stats["sum_v"]
        total_sum_sq += ds.stats["sum_sq"]
        total_count += ds.stats["count"]

    if total_count == 0:
        return 0.0, 1.0

    mean = total_sum / total_count
    std = float(np.sqrt(max(0, (total_sum_sq / total_count) - (mean ** 2))))
    return float(mean), max(std, 1e-8)


def apply_normalization_to_datasets(train_datasets, test_datasets):
    """Compute normalization from training data only and apply to all datasets."""
    if not train_datasets:
        print("  WARNING: no training datasets for normalization, using defaults (0, 1)")
        mean, std = 0.0, 1.0
    else:
        mean, std = compute_normalization_stats(train_datasets)
        print(f"  Normalization (from training only): mean={mean:.6f}, std={std:.6f}")
    for ds in train_datasets:
        ds.set_normalization(mean, std)
    for ds in test_datasets:
        ds.set_normalization(mean, std)


def create_datasets(patients_to_process, skip_missing_class=True, stage="finetune"):
    all_datasets = {}
    for idx in patients_to_process:
        pid = PATIENTS[idx]
        all_datasets[pid] = {}
        patient_config = get_patient_config(idx)
        dataset_prefix = patient_config["output_prefix"]
        dataset_dir = Path("preprocessing") / "data" / dataset_prefix
        if not dataset_dir.exists():
            print(f"patient {pid}: no preprocessing directory, skipping")
            continue
        h5_files = sorted(os.listdir(dataset_dir))
        skipped = []
        for h5_filename in h5_files:
            if not h5_filename.endswith("_dataset.h5"):
                continue
            split_name = h5_filename.strip().removeprefix("s").removesuffix("_dataset.h5")
            h5_file = dataset_dir / h5_filename
            dataset = EEGDataset(str(h5_file), split=split_name, stage=stage)
            if skip_missing_class:
                class_dist = torch.bincount(dataset.labels)
                if 0 in class_dist:
                    skipped.append(split_name)
                    del dataset
                    gc.collect()
                    continue
            all_datasets[pid][split_name] = dataset
        if skipped:
            print(f"patient {pid}: skipped folds {skipped} (missing class)")
        print(f"patient {pid} valid splits: {sorted(all_datasets[pid].keys())}")
    return all_datasets

def get_datasets_lopo(patient_id, all_datasets):
    test_datasets = []
    train_datasets = []
    for pid in all_datasets:
        if pid == patient_id:
            test_datasets = list(all_datasets[pid].values())
        else:
            train_datasets = train_datasets + list(all_datasets[pid].values())
    apply_normalization_to_datasets(train_datasets, test_datasets)
    test_combined_dataset = ConcatDataset(test_datasets)
    train_combined_dataset = ConcatDataset(train_datasets)
    return test_combined_dataset, train_combined_dataset

def get_datasets_per_patient(fold_id, patient_id, all_datasets):
    patient_datasets = all_datasets[patient_id]
    test_datasets = []
    train_datasets = []
    for sid in patient_datasets:
        if sid == fold_id:
            test_datasets = [patient_datasets[sid]]
        else:
            train_datasets = train_datasets + [patient_datasets[sid]]
    if not train_datasets or not test_datasets:
        return None, None
    apply_normalization_to_datasets(train_datasets, test_datasets)
    test_combined_dataset = ConcatDataset(test_datasets)
    train_combined_dataset = ConcatDataset(train_datasets)
    return test_combined_dataset, train_combined_dataset

class EEGCNNTrainer:
    def __init__(self, patient_config: Dict, datasets, lopo=False, test_seizure_id=0):
        self.patient_id = patient_config["patient_id"]
        self.test_seizure_id = test_seizure_id

        dataset_prefix = patient_config["output_prefix"]
        self.lopo = lopo
        if lopo:
            self.model_dir = Path("model") / "LOPO" / dataset_prefix
        else:
            self.model_dir = Path("model") / "per_patient" / dataset_prefix / f"test_seizure_{test_seizure_id}"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_prefix = dataset_prefix

        print(f"Dataset: {self.dataset_prefix}")
        print(f"Patient: {self.patient_id}")

        self.device = self._get_device()
        print(f"Device: {self.device}")

        test_dataset = None
        train_dataset = None
        if lopo:
            test_dataset, train_dataset = get_datasets_lopo(self.patient_id, datasets)
        else:
            test_dataset, train_dataset = get_datasets_per_patient(self.test_seizure_id,
                                                                   self.patient_id,
                                                                   datasets)
        if test_dataset is None or train_dataset is None:
            raise ValueError(
                f"Insufficient data for {self.patient_id} fold {test_seizure_id}: "
                f"need at least 1 test fold and 1 train fold with both classes present"
            )

        # Load Data
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=SEQUENCE_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(self.device.type == "cuda"),
        )
        self.val_loader = DataLoader(
            test_dataset,
            batch_size=SEQUENCE_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(self.device.type == "cuda"),
        )

        # Initialize Conv-GRU Model
        self.model = ConvGRUModel(
            num_input_channels=18,
            num_classes=2,
            embed_dim=CONV_EMBEDDING_DIM,
            gru_hidden=GRU_HIDDEN_DIM,
            gru_layers=GRU_NUM_LAYERS,
            dropout=GRU_DROPOUT,
        )

        # Load pretrained weights from the opposite group (no data leakage)
        my_group = get_pretrain_group(self.patient_id)
        pretrained_path = get_pretrain_path(1 - my_group)  # use OTHER group's encoder
        if not pretrained_path.exists():
            # Fall back to old single-file path
            pretrained_path = Path("model") / "pretrained_encoder.pth"
        if pretrained_path.exists() and not lopo:
            print(f"Loading pretrained encoder from {pretrained_path} (patient {self.patient_id} is group {my_group})")
            pretrained_state = torch.load(pretrained_path, map_location=self.device, weights_only=False)
            # Only load conv tower weights; GRU/norm/FC may have different dimensions
            conv_state = {k: v for k, v in pretrained_state.items() if k.startswith("conv_tower.")}
            self.model.load_state_dict(conv_state, strict=False)
            # Freeze conv tower (learned seizure spectral features from detection data).
            # Keep BiGRU, norm, and FC head trainable so the model
            # can learn preictal temporal buildup patterns per patient.
            for name, param in self.model.named_parameters():
                if name.startswith("conv_tower."):
                    param.requires_grad = False
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"  Conv tower frozen. Trainable: {trainable:,} / {total:,} params")

        self.model.to(self.device)

        # Class-weighted focal loss
        train_labels = self.train_loader.dataset
        if hasattr(train_labels, 'datasets'):
            # ConcatDataset
            all_labels = torch.cat([ds.labels for ds in train_labels.datasets])
        else:
            all_labels = train_labels.labels
        counts = torch.bincount(all_labels).float()
        class_weights = counts.sum() / (len(counts) * counts)
        self.criterion = FocalLoss(
            weight=class_weights.to(self.device), gamma=2.0, label_smoothing=0.05
        )

        # Only optimize trainable parameters
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=FINETUNING_LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
        )

        self.train_metrics_history = []
        self.val_metrics_history = []
        self.best_val_auc = 0.0

        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")


    def _get_device(self):
        if torch.cuda.is_available():
            print(f"CUDA detected: {torch.cuda.get_device_name()}")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print(f"MPS detected: Using Apple Silicon GPU acceleration")
            return torch.device("mps")
        print(f"Using CPU")
        return torch.device("cpu")

    def train_epoch(self, epoch):
        self.model.train()
        tracker = MetricsTracker()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Train")

        for x, labels in pbar:
            x, labels = x.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            tracker.update(preds, labels, probs)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        metrics = tracker.compute_metrics()
        metrics["loss"] = total_loss / len(self.train_loader)
        return metrics

    def validate_epoch(self, epoch, loader=None):
        self.model.eval()
        tracker = MetricsTracker()
        total_loss = 0.0

        target_loader = loader if loader is not None else self.val_loader
        desc = f"Epoch {epoch+1} Val" if isinstance(epoch, int) else f"{epoch}"
        pbar = tqdm(target_loader, desc=desc)

        with torch.no_grad():
            for x, labels in pbar:
                x, labels = x.to(self.device), labels.to(self.device)

                outputs = self.model(x)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
                tracker.update(preds, labels, probs)

        metrics = tracker.compute_metrics()
        metrics["loss"] = total_loss / len(target_loader)
        return metrics, tracker

    def save_model(self, epoch, train_metrics, val_metrics):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "best_val_auc": self.best_val_auc,
            "config": {
                "patient_id": self.patient_id,
                "architecture": "conv_gru",
            },
        }

        if val_metrics["auc_roc"] > self.best_val_auc:
            self.best_val_auc = val_metrics["auc_roc"]
            torch.save(checkpoint, self.model_dir / "best_model.pth")
            print(f"  New Best AUC: {self.best_val_auc:.4f}")

    def save_metrics(self):
        metrics_data = {
            "train_metrics": self.train_metrics_history,
            "val_metrics": self.val_metrics_history,
            "config": {
                "patient_id": self.patient_id,
                "epochs": TRAINING_EPOCHS,
                "batch_size": SEQUENCE_BATCH_SIZE,
                "lr": LEARNING_RATE,
                "architecture": "conv_gru",
            },
            "training_info": {
                "device": str(self.device),
                "timestamp": datetime.now().isoformat(),
            },
        }
        with open(self.model_dir / "training_metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2)

    def plot_training_curves(self):
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Patient {self.patient_id} Conv-GRU Training", fontsize=16)

        epochs = range(1, len(self.train_metrics_history) + 1)
        metrics = ["loss", "accuracy", "precision", "recall", "f1", "auc_roc"]

        for idx, m in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            ax.plot(
                epochs, [x[m] for x in self.train_metrics_history], "o-", label="Train"
            )
            ax.plot(epochs, [x[m] for x in self.val_metrics_history], "s-", label="Val")
            ax.set_title(m.upper())
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.model_dir / "training_curves.png")
        plt.close()

    def train(self):
        print("=" * 60)
        if self.lopo:
            print(f"STARTING TRAINING: LOPO {self.patient_id}")
        else:
            print(f"STARTING TRAINING: {self.patient_id} test seizure {self.test_seizure_id}")
        print("=" * 60)

        start_time = time.time()
        early_stopping_patience = 10
        epochs_no_improve = 0

        for epoch in range(TRAINING_EPOCHS):
            train_m = self.train_epoch(epoch)
            val_m, _ = self.validate_epoch(epoch)
            self.scheduler.step(val_m["auc_roc"])

            improved = val_m["auc_roc"] > self.best_val_auc
            self.save_model(epoch, train_m, val_m)
            self.train_metrics_history.append(train_m)
            self.val_metrics_history.append(val_m)

            if improved:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(f"Epoch {epoch+1}:")
            print(
                f"  Train | Loss: {train_m['loss']:.4f} | AUC: {train_m['auc_roc']:.4f} | Acc: {train_m['accuracy']:.4f} | Prec: {train_m['precision']:.4f} | Rec: {train_m['recall']:.4f} | F1: {train_m['f1']:.4f}"
            )
            print(
                f"  Val   | Loss: {val_m['loss']:.4f}   | AUC: {val_m['auc_roc']:.4f}   | Acc: {val_m['accuracy']:.4f}   | Prec: {val_m['precision']:.4f}   | Rec: {val_m['recall']:.4f}   | F1: {val_m['f1']:.4f}"
            )
            print("-" * 60)

            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping: no val AUC improvement for {early_stopping_patience} consecutive epochs.")
                break

        self.save_metrics()
        self.plot_training_curves()
        self.tune_best_model_threshold()
        print(f"Total time: {(time.time() - start_time)/60:.1f} min")

    def tune_best_model_threshold(self):
        best_path = self.model_dir / "best_model.pth"
        if not best_path.exists():
            return

        print(f"\nPerforming final threshold tuning on best model (using Training Set)...")
        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        val_metrics, tracker = self.validate_epoch(
            "Final Tuning (Train Set)", loader=self.train_loader
        )

        optimal_threshold = val_metrics["optimal_threshold"]

        all_probs = np.array(tracker.probabilities)
        all_labels = np.array(tracker.true_labels)
        min_len = min(len(all_probs), len(all_labels))
        all_probs = all_probs[:min_len]
        all_labels = all_labels[:min_len]
        optimized_preds = (all_probs >= optimal_threshold).astype(int)
        best_window, best_count = optimize_running_average(optimized_preds, all_labels)

        if "config" not in checkpoint:
            checkpoint["config"] = {}

        checkpoint["config"]["optimal_threshold"] = optimal_threshold
        checkpoint["config"]["smoothing_window"] = best_window
        checkpoint["config"]["smoothing_count"] = best_count
        checkpoint["config"]["stage"] = "finetune"

        torch.save(checkpoint, best_path)
        print(f"  Best model checkpoint updated:")
        print(f"   - Optimal Threshold: {optimal_threshold:.4f}")
        print(f"   - Smoothing Window: {best_window}")
        print(f"   - Smoothing Count: {best_count}")


def get_pretrain_group(patient_id):
    """Which group a patient belongs to. Even index → 0, odd → 1."""
    return PATIENTS.index(patient_id) % 2


def get_pretrain_path(group_id):
    """Path to pretrained encoder for a group."""
    return Path("model") / f"pretrained_encoder_group{group_id}.pth"


def pretrain():
    """Cross-patient pretraining on DETECTION data with two-group split.

    Uses ictal vs interictal (ground-truth seizure labels) to train the
    conv tower to recognize abnormal EEG spectral patterns. Detection
    labels are clinician-annotated, unlike prediction labels which depend
    on an estimated preictal boundary.

    Group 0: even-indexed patients, Group 1: odd-indexed patients.
    Trains two encoders — each on one group. When fine-tuning a patient
    in group X, the encoder pretrained on group (1-X) is loaded, so the
    encoder has never seen any of the fine-tuning patient's data.
    """
    print("=" * 60)
    print("CROSS-PATIENT PRETRAINING ON DETECTION DATA (two-group split)")
    print("=" * 60)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    group_patients = {0: [], 1: []}
    for i, pid in enumerate(PATIENTS):
        group_patients[i % 2].append(pid)

    print(f"Group 0 ({len(group_patients[0])} patients): {group_patients[0]}")
    print(f"Group 1 ({len(group_patients[1])} patients): {group_patients[1]}")

    for group_id in [0, 1]:
        save_path = get_pretrain_path(group_id)
        if save_path.exists():
            print(f"\nGroup {group_id} encoder already exists at {save_path}, skipping.")
            continue

        train_patients = group_patients[group_id]
        print(f"\n{'='*60}")
        print(f"Training encoder on group {group_id}: {train_patients}")
        print(f"(Will be used for patients in group {1 - group_id})")
        print(f"{'='*60}")

        datasets = []
        for patient_id in train_patients:
            dataset_dir = Path("preprocessing") / "data" / patient_id
            if not dataset_dir.exists():
                continue
            for h5_file in dataset_dir.glob("s*_dataset.h5"):
                datasets.append(EEGDataset(str(h5_file), split=f"pretrain ({patient_id})", augment=True, stage="pretrain"))

        if not datasets:
            print(f"No datasets for group {group_id}!")
            continue

        # All pretrain data is training data, so compute stats from all of it
        mean, std = compute_normalization_stats(datasets)
        print(f"  Pretrain normalization: mean={mean:.6f}, std={std:.6f}")
        for ds in datasets:
            ds.set_normalization(mean, std)

        combined = ConcatDataset(datasets)
        print(f"Training data: {len(combined)} samples from {len(datasets)} splits")

        loader = DataLoader(combined, batch_size=SEQUENCE_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        model = ConvGRUModel(
            num_input_channels=18, num_classes=2,
            embed_dim=CONV_EMBEDDING_DIM, gru_hidden=GRU_HIDDEN_DIM,
            gru_layers=GRU_NUM_LAYERS, dropout=GRU_DROPOUT,
        )
        model.to(device)
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

        all_labels = torch.cat([ds.labels for ds in datasets])
        counts = torch.bincount(all_labels).float()
        class_weights = counts.sum() / (len(counts) * counts)

        criterion = FocalLoss(weight=class_weights.to(device), gamma=2.0, label_smoothing=0.05)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

        for epoch in range(PRETRAINING_EPOCHS):
            model.train()
            total_loss = 0.0
            pbar = tqdm(loader, desc=f"Group {group_id} Epoch {epoch+1}/{PRETRAINING_EPOCHS}")

            for x, labels in pbar:
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)
            print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Group {group_id} encoder saved to {save_path}")

        del model, optimizer, scheduler, criterion, loader, combined, datasets
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="EEG Conv-GRU Training")
    parser.add_argument(
        "--pretrain", action="store_true",
        help="Run cross-patient pretraining instead of per-patient fine-tuning"
    )
    args = parser.parse_args()

    if args.pretrain:
        pretrain()
        return

    n_patients = len(PATIENTS)

    if LOPO_FOLD_ID is None:
        # per patient
        if PATIENT_INDEX is None:
            patients_to_process = list(range(n_patients))
        else:
            patients_to_process = [PATIENT_INDEX]

        for current_idx in patients_to_process:
            patient_id = PATIENTS[current_idx]
            if patient_id in PRETRAIN_ONLY_PATIENTS:
                print(f"Skipping {patient_id}: pretrain-only patient")
                continue
            patient_config = get_patient_config(current_idx)

            all_datasets = create_datasets([current_idx], skip_missing_class=True, stage="finetune")
            valid_folds = list(all_datasets[patient_id].keys())
            if len(valid_folds) < 2:
                print(f"Skipping {patient_id}: need at least 2 valid folds, got {len(valid_folds)}")
                continue
            test_seizures = valid_folds
            if TEST_SEIZURE is not None:
                test_seizures = [TEST_SEIZURE]
            print("patient:", patient_id, "seizures:", test_seizures)

            for sid in test_seizures:
                try:
                    gc.collect()
                    EEGCNNTrainer(patient_config, datasets=all_datasets, test_seizure_id=sid).train()
                except Exception as e:
                    print(f"Error {patient_config['patient_id']} seizure {sid}: {e}")
                    import traceback

                    traceback.print_exc()
    else:
        patients_to_process = list(range(n_patients))
        assert(LOPO_FOLD_ID < n_patients)

        all_datasets = create_datasets(patients_to_process, stage="finetune")

        patient_config = get_patient_config(LOPO_FOLD_ID)
        try:
            gc.collect()
            EEGCNNTrainer(patient_config, datasets=all_datasets,
                          lopo=True).train()
        except Exception as e:
            print(f"Error LOPO {patient_config['patient_id']} : {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
