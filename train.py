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
    """Single-Stream Dataset with lazy HDF5 loading for large 30-min windows."""

    def __init__(self, h5_file_path, split="train", augment=False):
        self.h5_file_path = h5_file_path
        self.split = split
        self.h5_file = None
        self.augment = augment and (split == "train")

        with h5py.File(h5_file_path, "r") as f:
            if "spectrograms_phase" in f or "spectrograms_amp" in f:
                raise ValueError(
                    f"HDF5 file {h5_file_path} uses old dual-stream format. "
                    "Re-run data_preprocessing.py to generate single-stream datasets."
                )
            self.labels = torch.LongTensor(f["labels"][:])
            self.length = len(self.labels)

            if "metadata" in f:
                self.metadata = dict(f["metadata"].attrs)

        print(f"Loaded {self.split} dataset: {self.length} samples")
        print(f"  - Class Dist: {torch.bincount(self.labels)}")
        if self.augment:
            print(f"  - Augmentation: ENABLED")

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
        x = torch.FloatTensor(self.h5_file["spectrograms"][idx])
        if self.augment:
            x = self._apply_augmentation(x)
        return x, self.labels[idx]

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()


class ConvTower(nn.Module):
    """Conv tower that extracts a fixed-size embedding from a single spectrogram segment.

    Input:  (batch, 18, 128, 9)
    Output: (batch, embed_dim)
    """

    def __init__(self, in_channels=18, embed_dim=128):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
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


class ConvTransformerModel(nn.Module):
    """Single-stream Conv-Transformer for EEG sequence classification.

    Conv tower tokenizes each 5s segment into an embedding, then a Transformer
    attends across all time positions to classify the full window.
    """
    Convolutional Fusion Module:
    1. Extracts Phase/Amp ribbons over time.
    2. Fuses them using 1x1 Conv (coupling detection at each time step).
    3. Pools time dimension ONLY after fusion.
    """

    def __init__(self, num_input_channels=18, embedding_dim=256):
        super().__init__()

        # Phase stream uses Sin/Cos (2 channels per EEG channel)
        self.phase_tower = CompactEEGCNN(num_input_channels * 2, output_channels=64)
        # Amp stream uses raw power (1 channel per EEG channel)
        self.amp_tower = CompactEEGCNN(num_input_channels, output_channels=64)

        # Fusion is now a 1x1 Convolution across time steps
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(64 + 64, embedding_dim, kernel_size=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        # Global Time Pooling (Happens AFTER fusion)
        self.gap_time = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_phase, x_amp):
        # 1. Extract Features (Time preserved)
        # Shape: (Batch, 64, 1, Time)
        feat_p = self.phase_tower(x_phase)
        feat_a = self.amp_tower(x_amp)

        # 2. Concatenate along Channels
        # Shape: (Batch, 128, 1, Time)
        combined = torch.cat((feat_p, feat_a), dim=1)

        # 3. Fuse (Pixel-wise coupling detection)
        # The Conv1x1 acts as a learnable "Dot Product" at each time step
        # Shape: (Batch, embedding_dim, 1, Time)
        fused_map = self.fusion_conv(combined)

        # 4. Now average over time
        # Shape: (Batch, embedding_dim, 1, 1)
        out = self.gap_time(fused_map)

        return out.view(out.size(0), -1)


class CNN_LSTM_Hybrid_Dual(nn.Module):
    """Dual-Stream Hybrid Model: Dual CNN -> Fusion -> BiLSTM -> Attention"""

    def __init__(
        self,
        num_input_channels=18,
        num_classes=2,
        sequence_length=SEQUENCE_LENGTH,
        embed_dim=CONV_EMBEDDING_DIM,
        num_layers=TRANSFORMER_NUM_LAYERS,
        num_heads=TRANSFORMER_NUM_HEADS,
        ffn_dim=TRANSFORMER_FFN_DIM,
        dropout=TRANSFORMER_DROPOUT,
        use_cls_token=USE_CLS_TOKEN,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        # Per-segment feature extractor
        self.conv_tower = ConvTower(in_channels=num_input_channels, embed_dim=embed_dim)

        # 1. Dual-Stream Encoder
        self.encoder = DualStreamSpectrogramEncoder(
            num_input_channels=num_input_channels, embedding_dim=fusion_dim
        )

        # 2. BiLSTM
        self.lstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=False,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Post-transformer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x_phase: (batch, seq, channels_phase, h_phase, w)
            x_amp:   (batch, seq, channels_amp, h_amp, w)
        """
        batch_size, seq_len, c_p, h_p, w = x_phase.shape
        _, _, c_a, h_a, _ = x_amp.shape

        # Flatten sequence dimension for CNN processing
        # (Batch * Seq, C, H, W)
        x_phase_flat = x_phase.view(batch_size * seq_len, c_p, h_p, w)
        x_amp_flat = x_amp.view(batch_size * seq_len, c_a, h_a, w)

        # Conv tower: (B*S, embed_dim)
        embeddings = self.conv_tower(x_flat)

        # Reshape to sequence: (B, S, embed_dim)
        embeddings = embeddings.view(B, S, self.embed_dim)

        # Add CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            embeddings = torch.cat([cls_tokens, embeddings], dim=1)

        # Add positional encoding
        embeddings = embeddings + self.pos_embedding

        # Transformer
        out = self.transformer(embeddings)
        out = self.norm(out)

        # Pooling
        if self.use_cls_token:
            pooled = out[:, 0]  # CLS token output
        else:
            pooled = out.mean(dim=1)  # Mean pooling

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

    for window_size in range(5, max_window + 1):
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

def create_datasets(patients_to_process, skip_missing_class=True):
    all_datasets = {}
    for idx in patients_to_process:
        pid = PATIENTS[idx]
        all_datasets[pid] = {}
        patient_config = get_patient_config(idx)
        dataset_prefix = patient_config["output_prefix"]
        dataset_dir = Path("preprocessing") / "data" / dataset_prefix
        h5_files = os.listdir(dataset_dir)
        for h5_filename in h5_files:
            split_name = h5_filename.strip().removeprefix("s").removesuffix("_dataset.h5")
            h5_file = dataset_dir / h5_filename
            dataset = EEGDataset(str(h5_file), split=split_name)
            if skip_missing_class:
                class_dist = torch.bincount(dataset.labels)
                if 0 in class_dist:
                    print(f"skipping patient {pid} seizure {split_name}: missing label")
                    del dataset
                    gc.collect()
                    continue
            all_datasets[pid][split_name] = dataset
        print(f"patient {pid} splits: {all_datasets[pid].keys()}")
    return all_datasets
    
def get_datasets_lopo(patient_id, all_datasets):
    test_datasets = []
    train_datasets = []
    for pid in all_datasets:
        if pid == patient_id:
            test_datasets = list(all_datasets[pid].values())
        else:
            train_datasets = train_datasets + list(all_datasets[pid].values())
    test_combined_dataset = ConcatDataset(test_datasets)
    train_combined_dataset = ConcatDataset(train_datasets)
    return test_combined_dataset, train_combined_dataset
def get_datasets_per_patient(fold_id, patient_id, all_datasets):
    patient_datasets = all_datasets[patient_id]
    test_datasets = []
    train_datasets = []
    for sid in  patient_datasets:
        if sid == fold_id:
            test_datasets = [patient_datasets[sid]]
        else:
            train_datasets = train_datasets + [patient_datasets[sid]]
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
        self.dataset_dir = Path("preprocessing") / "data" / self.dataset_prefix

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

        val_h5 = self.dataset_dir / "val_dataset.h5"
        if val_h5.exists():
            self.val_loader = self._create_dataloader("val")
        else:
            print(
                f"  WARNING: No val split found for {self.patient_id}. "
                "Falling back to test set for monitoring."
            )
            self.val_loader = self._create_dataloader("test")

        # Initialize Conv-Transformer Model
        self.model = ConvTransformerModel(
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

        # Load pretrained weights if available
        pretrained_path = Path("model") / "pretrained_encoder.pth"
        if pretrained_path.exists():
            print(f"Loading pretrained encoder from {pretrained_path}")
            self.model.load_state_dict(
                torch.load(pretrained_path, map_location=self.device, weights_only=False)
            )

        self.model.to(self.device)

        # Class-weighted focal loss
        train_labels = self.train_loader.dataset.labels
        counts = torch.bincount(train_labels).float()
        class_weights = counts.sum() / (len(counts) * counts)
        self.criterion = FocalLoss(
            weight=class_weights.to(self.device), gamma=2.0, label_smoothing=0.05
        )

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=FINETUNING_LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
        )

        self.train_metrics_history = []
        self.val_metrics_history = []
        self.best_val_auc = 0.0

        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")


    @property
    def positive_label(self):
        return "preictal" if TASK_MODE == "prediction" else "ictal"

    def _get_device(self):
        if torch.cuda.is_available():
            print(f"CUDA detected: {torch.cuda.get_device_name()}")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print(f"MPS detected: Using Apple Silicon GPU acceleration")
            return torch.device("mps")
        print(f"Using CPU")
        return torch.device("cpu")

    # def _create_dataloader(self, split):
    #     h5_file = self.dataset_dir / f"{split}_dataset.h5"
    #     if not h5_file.exists():
    #         raise FileNotFoundError(f"Missing {h5_file}")

    #     dataset = EEGDataset(str(h5_file), split=split)
    #     return DataLoader(
    #         dataset,
    #         batch_size=SEQUENCE_BATCH_SIZE,
    #         shuffle=(split == "train"),
    #         num_workers=NUM_WORKERS,
    #         pin_memory=(self.device.type == "cuda"),
    #     )

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
                "architecture": "conv_transformer",
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
                "architecture": "conv_transformer",
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
        fig.suptitle(f"Patient {self.patient_id} Conv-Transformer Training", fontsize=16)

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
            print(f"STARTING DUAL-STREAM TRAINING: LOPO {self.patient_id}")
        else:
            print(f"STARTING DUAL-STREAM TRAINING: {self.patient_id} test seizure {self.test_seizure_id}")
        print("=" * 60)

        start_time = time.time()
        early_stopping_patience = 5
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

        print(f"\nPerforming final threshold tuning on best model (using Validation Set)...")
        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        val_metrics, tracker = self.validate_epoch(
            "Final Tuning (Val Set)", loader=self.val_loader
        )

        optimal_threshold = val_metrics["optimal_threshold"]

        all_probs = np.array(tracker.probabilities)
        all_labels = np.array(tracker.true_labels)
        optimized_preds = (all_probs >= optimal_threshold).astype(int)
        best_window, best_count = optimize_running_average(optimized_preds, all_labels)

        if "config" not in checkpoint:
            checkpoint["config"] = {}

        checkpoint["config"]["optimal_threshold"] = optimal_threshold
        checkpoint["config"]["smoothing_window"] = best_window
        checkpoint["config"]["smoothing_count"] = best_count
        checkpoint["config"]["task_mode"] = TASK_MODE

        torch.save(checkpoint, best_path)
        print(f"  Best model checkpoint updated:")
        print(f"   - Optimal Threshold: {optimal_threshold:.4f}")
        print(f"   - Smoothing Window: {best_window}")
        print(f"   - Smoothing Count: {best_count}")


def pretrain():
    """Cross-patient pretraining: train shared encoder on ALL patients."""
    print("=" * 60)
    print("CROSS-PATIENT PRETRAINING")
    print("=" * 60)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load train datasets from all patients
    all_datasets = []
    for patient_id in PATIENTS:
        h5_path = Path("preprocessing") / "data" / patient_id / "train_dataset.h5"
        if h5_path.exists():
            all_datasets.append(EEGDataset(str(h5_path), split=f"train ({patient_id})", augment=True))
        else:
            print(f"  Skipping {patient_id}: no train dataset found")

    if not all_datasets:
        print("No datasets found for pretraining!")
        return

    combined = ConcatDataset(all_datasets)
    print(f"\nCombined pretraining dataset: {len(combined)} samples from {len(all_datasets)} patients")

    pretrain_loader = DataLoader(
        combined,
        batch_size=SEQUENCE_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    # Initialize model
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
    model.to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compute class weights from combined labels
    all_labels = torch.cat([ds.labels for ds in all_datasets])
    counts = torch.bincount(all_labels).float()
    class_weights = counts.sum() / (len(counts) * counts)

    criterion = FocalLoss(
        weight=class_weights.to(device), gamma=2.0, label_smoothing=0.05
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    # Train
    for epoch in range(PRETRAINING_EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(pretrain_loader, desc=f"Pretrain Epoch {epoch+1}/{PRETRAINING_EPOCHS}")

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

        avg_loss = total_loss / len(pretrain_loader)
        scheduler.step(avg_loss)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Save pretrained weights
    save_dir = Path("model")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "pretrained_encoder.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nPretrained encoder saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="EEG Conv-Transformer Training")
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
            patient_config = get_patient_config(current_idx)

            all_datasets = create_datasets([current_idx], skip_missing_class=True)
            test_seizures = all_datasets[patient_id].keys()
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

        all_datasets = create_datasets(patients_to_process)

        patient_config = get_patient_config(LOPO_FOLD_ID)
        print("HERE")
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
