import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
)
from tqdm import tqdm
import time
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from data_segmentation_helpers.config import (
    TASK_MODE,
    SEQUENCE_LENGTH,
    SEQUENCE_BATCH_SIZE,
    NUM_WORKERS,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
    TRAINING_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENTS,
    PATIENT_INDEX,
    get_patient_config,
)


class EEGDataset(Dataset):
    """Dual-Stream Dataset for Phase (Low-Freq) and Amplitude (High-Freq) inputs"""

    def __init__(self, h5_file_path, split="train"):
        self.h5_file_path = h5_file_path
        self.split = split

        # Load data into memory (Phase and Amp streams)
        with h5py.File(h5_file_path, "r") as f:
            self.phase_data = torch.FloatTensor(f["spectrograms_phase"][:])
            self.amp_data = torch.FloatTensor(f["spectrograms_amp"][:])
            self.labels = torch.LongTensor(f["labels"][:])
            self.patient_ids = [pid.decode("utf-8") for pid in f["patient_ids"][:]]

            if "metadata" in f:
                self.metadata = dict(f["metadata"].attrs)

        print(f"Loaded {self.split} dataset: {len(self.labels)} samples")
        print(f"  - Phase Shape: {self.phase_data.shape} (Time/Timing)")
        print(f"  - Amp Shape:   {self.amp_data.shape} (Power/Energy)")
        print(f"  - Class Dist:  {torch.bincount(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns:
            x_phase: (sequence_length, channels, freq_phase, time)
            x_amp:   (sequence_length, channels, freq_amp, time)
            label:   int
        """
        return self.phase_data[idx], self.amp_data[idx], self.labels[idx]


class CompactEEGCNN(nn.Module):
    """
    A lightweight CNN backbone that preserves Temporal Resolution.
    Pools Frequency (Height) to 1, but keeps Time (Width) intact.
    """

    def __init__(self, in_channels=18, output_channels=64):
        super(CompactEEGCNN, self).__init__()

        # Block 1: Capture basic patterns
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),  # Regularization
            nn.MaxPool2d(2),  # Reduces Time and Freq by 2
        )

        # Block 2: Spatial compositions
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),  # Regularization
            nn.MaxPool2d(2),  # Reduces Time and Freq by 2
        )

        # Block 3: Complex features (Preserve Time)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),  # Regularization
            # No Pooling here (or use MaxPool2d((2, 1)) to pool Freq only)
        )

        # CRITICAL: Adaptive Pool Height (Freq) to 1, Keep Width (Time)
        # This creates a "ribbon" of features over time
        self.gap_freq = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Output shape: (Batch, Channels, 1, Time)
        x = self.gap_freq(x)
        return x


class DualStreamSpectrogramEncoder(nn.Module):
    """
    Convolutional Fusion Module:
    1. Extracts Phase/Amp ribbons over time.
    2. Fuses them using 1x1 Conv (coupling detection at each time step).
    3. Pools time dimension ONLY after fusion.
    """

    def __init__(self, in_channels=18, embedding_dim=256):
        super().__init__()

        self.phase_tower = CompactEEGCNN(in_channels, output_channels=64)
        self.amp_tower = CompactEEGCNN(in_channels, output_channels=64)

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
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        lstm_num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
    ):
        super().__init__()

        fusion_dim = 256  # Output size of the DualStreamEncoder

        # 1. Dual-Stream Encoder
        self.encoder = DualStreamSpectrogramEncoder(
            in_channels=num_input_channels, embedding_dim=fusion_dim
        )

        # 2. BiLSTM
        self.lstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True,
        )

        # 3. Attention
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False),
        )

        # 4. Classifier
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_phase, x_amp):
        """
        Args:
            x_phase: (batch, seq, channels, h_phase, w)
            x_amp:   (batch, seq, channels, h_amp, w)
        """
        batch_size, seq_len, c, h_p, w = x_phase.shape
        _, _, _, h_a, _ = x_amp.shape

        # Flatten sequence dimension for CNN processing
        # (Batch * Seq, C, H, W)
        x_phase_flat = x_phase.view(batch_size * seq_len, c, h_p, w)
        x_amp_flat = x_amp.view(batch_size * seq_len, c, h_a, w)

        # Dual-Stream Encoding
        features = self.encoder(x_phase_flat, x_amp_flat)  # (Batch*Seq, FusionDim)

        # Reshape back to sequence
        features = features.view(batch_size, seq_len, -1)

        # LSTM Processing
        lstm_out, _ = self.lstm(features)

        # Attention Pooling
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        return self.fc(context)


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

        # Handle edge case where only one class is present in batch
        if len(np.unique(true_np)) > 1:
            auc = roc_auc_score(true_np, prob_np)
        else:
            auc = 0.5

        return {
            "accuracy": accuracy_score(true_np, pred_np),
            "precision": precision_score(
                true_np, pred_np, average="binary", zero_division=0
            ),
            "recall": recall_score(true_np, pred_np, average="binary", zero_division=0),
            "f1": f1_score(true_np, pred_np, average="binary", zero_division=0),
            "auc_roc": auc,
        }


class EEGCNNTrainer:
    def __init__(self, patient_config: Dict):
        self.patient_id = patient_config["patient_id"]
        dataset_prefix = patient_config["output_prefix"]

        self.model_dir = Path("model") / dataset_prefix
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_prefix = dataset_prefix
        self.dataset_dir = Path("preprocessing") / "data" / self.dataset_prefix

        print(f"Dataset: {self.dataset_prefix}")
        print(f"Patient: {self.patient_id}")

        self.device = self._get_device()
        print(f"Device: {self.device}")

        # Load Data
        self.train_loader = self._create_dataloader("train")
        self.val_loader = self._create_dataloader("test")

        # Initialize Dual-Stream Model
        self.model = CNN_LSTM_Hybrid_Dual(
            num_input_channels=18,
            num_classes=2,
            lstm_hidden_dim=LSTM_HIDDEN_DIM,
            lstm_num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT,
        )
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )

        self.train_metrics_history = []
        self.val_metrics_history = []
        self.best_val_auc = 0.0

        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    @property
    def positive_label(self):
        return "preictal" if TASK_MODE == "prediction" else "ictal"

    def _get_device_name(self):
        if self.device.type == "cuda":
            return torch.cuda.get_device_name()
        elif self.device.type == "mps":
            return "Apple Silicon GPU (MPS)"
        return "CPU"

    def _get_device(self):
        if torch.cuda.is_available():
            print(f"CUDA detected: {torch.cuda.get_device_name()}")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print(f"MPS detected: Using Apple Silicon GPU acceleration")
            return torch.device("mps")
        print(f"Using CPU")
        return torch.device("cpu")

    def _create_dataloader(self, split):
        h5_file = self.dataset_dir / f"{split}_dataset.h5"
        if not h5_file.exists():
            raise FileNotFoundError(f"Missing {h5_file}")

        dataset = EEGDataset(str(h5_file), split=split)
        return DataLoader(
            dataset,
            batch_size=SEQUENCE_BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=NUM_WORKERS,
            pin_memory=(self.device.type == "cuda"),
        )

    def train_epoch(self, epoch):
        self.model.train()
        tracker = MetricsTracker()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Train")

        for x_phase, x_amp, labels in pbar:
            x_phase, x_amp, labels = (
                x_phase.to(self.device),
                x_amp.to(self.device),
                labels.to(self.device),
            )

            self.optimizer.zero_grad()
            outputs = self.model(x_phase, x_amp)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            tracker.update(preds, labels, probs)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        metrics = tracker.compute_metrics()
        metrics["loss"] = total_loss / len(self.train_loader)
        return metrics

    def validate_epoch(self, epoch):
        self.model.eval()
        tracker = MetricsTracker()
        total_loss = 0.0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} Val")

        with torch.no_grad():
            for x_phase, x_amp, labels in pbar:
                x_phase, x_amp, labels = (
                    x_phase.to(self.device),
                    x_amp.to(self.device),
                    labels.to(self.device),
                )

                outputs = self.model(x_phase, x_amp)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
                tracker.update(preds, labels, probs)

        metrics = tracker.compute_metrics()
        metrics["loss"] = total_loss / len(self.val_loader)
        return metrics

    def save_model(self, epoch, train_metrics, val_metrics):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "best_val_auc": self.best_val_auc,
            "config": {"patient_id": self.patient_id},
        }

        # Only save if we have a new best model
        if val_metrics["auc_roc"] > self.best_val_auc:
            self.best_val_auc = val_metrics["auc_roc"]
            torch.save(checkpoint, self.model_dir / "best_model.pth")
            print(f"⭐ New Best AUC: {self.best_val_auc:.4f}")

    def save_metrics(self):
        metrics_data = {
            "train_metrics": self.train_metrics_history,
            "val_metrics": self.val_metrics_history,
            "config": {
                "patient_id": self.patient_id,
                "epochs": TRAINING_EPOCHS,
                "batch_size": SEQUENCE_BATCH_SIZE,
                "lr": LEARNING_RATE,
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
        fig.suptitle(f"Patient {self.patient_id} Dual-Stream Training", fontsize=16)

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
        print(f"STARTING DUAL-STREAM TRAINING: {self.patient_id}")
        print("=" * 60)

        start_time = time.time()
        for epoch in range(TRAINING_EPOCHS):
            train_m = self.train_epoch(epoch)
            val_m = self.validate_epoch(epoch)
            self.scheduler.step()

            self.save_model(epoch, train_m, val_m)
            self.train_metrics_history.append(train_m)
            self.val_metrics_history.append(val_m)

            print(f"Epoch {epoch+1}:")
            print(
                f"  Train | Loss: {train_m['loss']:.4f} | AUC: {train_m['auc_roc']:.4f} | Acc: {train_m['accuracy']:.4f} | Prec: {train_m['precision']:.4f} | Rec: {train_m['recall']:.4f} | F1: {train_m['f1']:.4f}"
            )
            print(
                f"  Val   | Loss: {val_m['loss']:.4f}   | AUC: {val_m['auc_roc']:.4f}   | Acc: {val_m['accuracy']:.4f}   | Prec: {val_m['precision']:.4f}   | Rec: {val_m['recall']:.4f}   | F1: {val_m['f1']:.4f}"
            )
            print("-" * 60)

        self.save_metrics()
        self.plot_training_curves()
        print(f"Total time: {(time.time() - start_time)/60:.1f} min")


def main():
    n_patients = len(PATIENTS)
    if PATIENT_INDEX is None:
        patients_to_process = list(range(n_patients))
    else:
        patients_to_process = [PATIENT_INDEX]

    for current_idx in patients_to_process:
        patient_config = get_patient_config(current_idx)
        try:
            EEGCNNTrainer(patient_config).train()
        except Exception as e:
            print(f"Error {patient_config['patient_id']}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
