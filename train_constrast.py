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

from data_segmentation_helpers.config import (
    TASK_MODE,
    SEQUENCE_LENGTH,
    SEQUENCE_BATCH_SIZE,
    NUM_WORKERS,
    CONV_EMBEDDING_DIM,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_NUM_HEADS,
    TRANSFORMER_FFN_DIM,
    TRANSFORMER_DROPOUT,
    USE_CLS_TOKEN,
    PRETRAINING_EPOCHS,
    TRAINING_EPOCHS,
    LEARNING_RATE,
    FINETUNING_LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENTS,
    PATIENT_INDEX,
    get_patient_config,
)


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
        """Apply SpecAugment-style augmentation.
        x: (sequence_length, channels, freq_bins, time_bins)
        """
        seq_len = x.shape[0]
        freq_bins = x.shape[2]

        # Time masking: zero out 1-2 random contiguous blocks of segments
        for _ in range(torch.randint(1, 3, (1,)).item()):
            if torch.rand(1).item() < 0.5:
                mask_len = torch.randint(1, max(2, seq_len // 8), (1,)).item()
                start = torch.randint(0, seq_len - mask_len + 1, (1,)).item()
                x[start : start + mask_len] = 0.0

        # Frequency masking: zero out 1-2 random frequency bands
        for _ in range(torch.randint(1, 3, (1,)).item()):
            if torch.rand(1).item() < 0.5:
                mask_len = torch.randint(1, max(2, freq_bins // 6), (1,)).item()
                start = torch.randint(0, freq_bins - mask_len + 1, (1,)).item()
                x[:, :, start : start + mask_len, :] = 0.0

        # Channel masking: zero out 1-3 random channels
        if torch.rand(1).item() < 0.3:
            n_channels = x.shape[1]
            n_mask = torch.randint(1, min(4, n_channels), (1,)).item()
            channels_to_mask = torch.randperm(n_channels)[:n_mask]
            x[:, channels_to_mask, :, :] = 0.0

        # Gaussian noise
        if torch.rand(1).item() < 0.5:
            noise_std = 0.1 * x.std()
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


class EEGChannelAttention(nn.Module):
    """Squeeze-excitation attention over EEG input channels (18 channels).

    Learns which EEG channels are most informative per sample.
    Input/Output: (batch, 18, freq, time)
    """

    def __init__(self, n_channels=18, reduction=4):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_channels, n_channels // reduction),
            nn.GELU(),
            nn.Linear(n_channels // reduction, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.attn(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ChannelSEBlock(nn.Module):
    """Squeeze-excitation attention over conv feature map channels.

    Learns which feature maps matter after conv processing.
    Input/Output: (batch, channels, H, W)
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class AttentionPooling(nn.Module):
    """Learnable attention pooling over sequence positions.

    Replaces CLS token — learns per-position weights and returns weighted sum.
    Input: (batch, seq_len, embed_dim) → Output: (batch, embed_dim)
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1, bias=False),
        )

    def forward(self, x):
        weights = self.attention(x)  # (B, S, 1)
        weights = torch.softmax(weights, dim=1)
        return (x * weights).sum(dim=1)  # (B, D)


class ConvTower(nn.Module):
    """Conv tower that extracts a fixed-size embedding from a single spectrogram segment.

    Includes EEG channel attention (input level) and SE block (feature level).
    Input:  (batch, 18, 128, 9)
    Output: (batch, embed_dim)
    """

    def __init__(self, in_channels=18, embed_dim=128):
        super().__init__()

        self.channel_attn = EEGChannelAttention(in_channels)

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

        self.se = ChannelSEBlock(embed_dim)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.se(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class ConvTransformerModel(nn.Module):
    """Single-stream Conv-Transformer for EEG sequence classification.

    Conv tower tokenizes each 5s segment into an embedding, then a Transformer
    attends across all time positions to classify the full window.
    """

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

        # CLS token / attention pooling and positional embeddings
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.pos_embedding = nn.Parameter(
                torch.randn(1, sequence_length + 1, embed_dim) * 0.02
            )
            self.attn_pool = None
        else:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, sequence_length, embed_dim) * 0.02
            )
            self.attn_pool = AttentionPooling(embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
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
            x: (batch, seq_len, channels, freq, time)
        Returns:
            (batch, num_classes)
        """
        B, S, C, H, W = x.shape

        # Flatten sequence for CNN: (B*S, C, H, W)
        x_flat = x.view(B * S, C, H, W)

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
        elif self.attn_pool is not None:
            pooled = self.attn_pool(out)  # Attention pooling
        else:
            pooled = out.mean(dim=1)  # Mean pooling fallback

        return self.fc(pooled)

    def encode(self, x):
        """Return pooled embedding before FC head (for contrastive learning)."""
        B, S, C, H, W = x.shape
        x_flat = x.view(B * S, C, H, W)
        embeddings = self.conv_tower(x_flat)
        embeddings = embeddings.view(B, S, self.embed_dim)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        embeddings = embeddings + self.pos_embedding
        out = self.transformer(embeddings)
        out = self.norm(out)
        if self.use_cls_token:
            return out[:, 0]
        elif self.attn_pool is not None:
            return self.attn_pool(out)
        else:
            return out.mean(dim=1)


# =============================================================================
# Contrastive Pretraining Components
# =============================================================================


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning. Discarded after pretraining."""

    def __init__(self, input_dim=128, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020).

    Pulls same-class embeddings together, pushes different-class apart.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: (2N, proj_dim) — concatenated projections from both views
            labels: (2N,) — duplicated labels
        """
        device = features.device
        batch_size = features.shape[0]

        features = F.normalize(features, dim=1)

        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature

        # Mask out self-similarity
        mask_self = torch.eye(batch_size, dtype=torch.bool, device=device)
        sim.masked_fill_(mask_self, -1e9)

        # Positive mask: same label, different sample
        labels_col = labels.unsqueeze(0)
        labels_row = labels.unsqueeze(1)
        pos_mask = (labels_col == labels_row) & ~mask_self

        # Log-softmax over non-self entries
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Average log-prob over positive pairs
        n_pos = pos_mask.sum(dim=1).clamp(min=1)
        loss = -(pos_mask * log_prob).sum(dim=1) / n_pos

        return loss.mean()


class ContrastiveEEGDataset(EEGDataset):
    """Returns two independently augmented views of each sample."""

    def __init__(self, h5_file_path, split="train"):
        super().__init__(h5_file_path, split=split, augment=False)

    def __getitem__(self, idx):
        self._open_h5()
        x = torch.FloatTensor(self.h5_file["spectrograms"][idx])
        view1 = self._apply_augmentation(x.clone())
        view2 = self._apply_augmentation(x.clone())
        return view1, view2, self.labels[idx]


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

    # Cap max window at sample count to prevent np.convolve from returning longer output
    effective_max = min(max_window, min_len)
    for window_size in range(5, effective_max + 1):
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


class EEGTrainer:
    def __init__(self, patient_config: Dict, pretrained_path=None, model_subdir=None):
        self.patient_id = patient_config["patient_id"]
        dataset_prefix = patient_config["output_prefix"]

        if model_subdir:
            self.model_dir = Path("model") / "contrastive" / model_subdir / dataset_prefix
        else:
            self.model_dir = Path("model") / "contrastive" / dataset_prefix
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_prefix = dataset_prefix
        self.dataset_dir = Path("preprocessing") / "data" / self.dataset_prefix

        print(f"Dataset: {self.dataset_prefix}")
        print(f"Patient: {self.patient_id}")

        self.device = self._get_device()
        print(f"Device: {self.device}")

        # Load Data
        self.train_loader = self._create_dataloader("train")

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
        if pretrained_path is None:
            pretrained_path = Path("model") / "contrastive" / "pretrained_encoder.pth"
        else:
            pretrained_path = Path(pretrained_path)
        # Store pretrained config directory (for temperature loading)
        self.pretrained_config_path = pretrained_path.parent / "pretrained_config.json"
        if pretrained_path.exists():
            print(f"Loading pretrained encoder from {pretrained_path}")
            self.model.load_state_dict(
                torch.load(pretrained_path, map_location=self.device, weights_only=False)
            )

            # Freeze conv tower — only fine-tune transformer + classifier head
            for param in self.model.conv_tower.parameters():
                param.requires_grad = False
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"  Conv tower frozen: {trainable:,}/{total:,} params trainable")

        self.model.to(self.device)

        # Class-weighted cross-entropy (better calibrated than FocalLoss for fine-tuning)
        train_labels = self.train_loader.dataset.labels
        counts = torch.bincount(train_labels).float()
        class_weights = counts.sum() / (len(counts) * counts)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device), label_smoothing=0.05
        )

        # Only optimize trainable parameters (transformer + head when conv tower is frozen)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=FINETUNING_LEARNING_RATE, weight_decay=WEIGHT_DECAY
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

    def _create_dataloader(self, split):
        h5_file = self.dataset_dir / f"{split}_dataset.h5"
        if not h5_file.exists():
            raise FileNotFoundError(f"Missing {h5_file}")

        dataset = EEGDataset(str(h5_file), split=split, augment=(split == "train"))
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
        print(f"STARTING CONV-TRANSFORMER TRAINING: {self.patient_id}")
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

        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)

        # Load temperature from pretrained config
        pretrained_config_path = self.pretrained_config_path
        if pretrained_config_path.exists():
            with open(pretrained_config_path, "r") as f:
                pretrained_config = json.load(f)
            temperature = pretrained_config.get("temperature", 1.0)
        else:
            temperature = 1.0

        # Tune threshold on per-patient val set (not leakage — val is separate from test)
        print(f"\nTuning threshold on per-patient val set (temperature={temperature:.2f})...")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        all_logits = []
        all_labels = []
        with torch.no_grad():
            for x, labels in self.val_loader:
                x, labels = x.to(self.device), labels.to(self.device)
                outputs = self.model(x)
                all_logits.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        logits_np = np.array(all_logits)
        labels_np = np.array(all_labels)

        # Apply temperature scaling to get calibrated probabilities
        scaled_logits = torch.FloatTensor(logits_np) / temperature
        all_probs = torch.softmax(scaled_logits, dim=1)[:, 1].numpy()

        if len(np.unique(labels_np)) > 1:
            fpr, tpr, thresholds = roc_curve(labels_np, all_probs)
            j_scores = tpr - fpr
            optimal_threshold = float(thresholds[np.argmax(j_scores)])
        else:
            optimal_threshold = 0.5

        optimized_preds = (all_probs >= optimal_threshold).astype(int)
        # Ensure same length (edge case with batching)
        min_len = min(len(optimized_preds), len(labels_np))
        optimized_preds = optimized_preds[:min_len]
        labels_np = labels_np[:min_len]
        best_window, best_count = optimize_running_average(optimized_preds, labels_np)

        if "config" not in checkpoint:
            checkpoint["config"] = {}

        checkpoint["config"]["optimal_threshold"] = optimal_threshold
        checkpoint["config"]["smoothing_window"] = best_window
        checkpoint["config"]["smoothing_count"] = best_count
        checkpoint["config"]["temperature"] = temperature
        checkpoint["config"]["task_mode"] = TASK_MODE

        torch.save(checkpoint, best_path)
        print(f"  Best model checkpoint updated:")
        print(f"   - Optimal Threshold: {optimal_threshold:.4f}")
        print(f"   - Smoothing Window: {best_window}")
        print(f"   - Smoothing Count: {best_count}")
        print(f"   - Temperature: {temperature:.2f}")


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def pretrain(exclude_patient=None, seed=None, save_suffix=None):
    """Supervised contrastive pretraining on ALL patients.

    Phase 1: SupCon loss with two augmented views per sample.
    Phase 2: Brief linear probe (train FC head) for downstream threshold tuning.

    Args:
        exclude_patient: If set, exclude this patient from pretraining (for LOPO).
        seed: Random seed for reproducibility (for ensemble).
        save_suffix: Suffix for save directory (e.g., "seed0" for ensemble).
    """
    CONTRASTIVE_TEMPERATURE = 0.07
    PROJECTION_DIM = 64
    LINEAR_PROBE_EPOCHS = 10

    print("=" * 60)
    if exclude_patient:
        print(f"CONTRASTIVE LOPO PRETRAINING (excluding {exclude_patient})")
    elif save_suffix:
        print(f"CONTRASTIVE ENSEMBLE PRETRAINING ({save_suffix}, seed={seed})")
    else:
        print("CONTRASTIVE CROSS-PATIENT PRETRAINING")
    print("=" * 60)

    if seed is not None:
        set_seed(seed)
        print(f"Random seed: {seed}")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load contrastive datasets (two augmented views per sample)
    contrastive_datasets = []
    for patient_id in PATIENTS:
        if patient_id == exclude_patient:
            print(f"  LOPO: Excluding {patient_id} from pretraining")
            continue
        h5_path = Path("preprocessing") / "data" / patient_id / "train_dataset.h5"
        if h5_path.exists():
            contrastive_datasets.append(
                ContrastiveEEGDataset(str(h5_path), split=f"train ({patient_id})")
            )
        else:
            print(f"  Skipping {patient_id}: no train dataset found")

    if not contrastive_datasets:
        print("No datasets found for pretraining!")
        return

    combined = ConcatDataset(contrastive_datasets)
    print(f"\nContrastive dataset: {len(combined)} samples from {len(contrastive_datasets)} patients")

    pretrain_loader = DataLoader(
        combined,
        batch_size=SEQUENCE_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,  # SupCon needs at least 2 samples per class in batch
    )

    # Initialize model + projection head
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

    proj_head = ProjectionHead(
        input_dim=CONV_EMBEDDING_DIM,
        hidden_dim=CONV_EMBEDDING_DIM,
        output_dim=PROJECTION_DIM,
    ).to(device)

    encoder_params = sum(p.numel() for p in model.parameters())
    proj_params = sum(p.numel() for p in proj_head.parameters())
    print(f"Encoder params: {encoder_params:,}  |  Projection head: {proj_params:,}")

    criterion = SupConLoss(temperature=CONTRASTIVE_TEMPERATURE)
    all_params = list(model.parameters()) + list(proj_head.parameters())
    optimizer = optim.Adam(all_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PRETRAINING_EPOCHS - 5, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[5]
    )

    # ===================== Phase 1: Contrastive Pretraining =====================
    print(f"\n--- Phase 1: Supervised Contrastive Learning ({PRETRAINING_EPOCHS} epochs) ---")
    for epoch in range(PRETRAINING_EPOCHS):
        model.train()
        proj_head.train()
        total_loss = 0.0
        pbar = tqdm(pretrain_loader, desc=f"SupCon Epoch {epoch+1}/{PRETRAINING_EPOCHS}")

        for view1, view2, labels in pbar:
            view1, view2 = view1.to(device), view2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Encode both views
            z1 = model.encode(view1)  # (B, embed_dim)
            z2 = model.encode(view2)  # (B, embed_dim)

            # Project
            p1 = proj_head(z1)  # (B, proj_dim)
            p2 = proj_head(z2)  # (B, proj_dim)

            # Concatenate views: (2B, proj_dim) and (2B,) labels
            features = torch.cat([p1, p2], dim=0)
            labels_2x = torch.cat([labels, labels], dim=0)

            loss = criterion(features, labels_2x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(pretrain_loader)
        scheduler.step()
        print(f"  Epoch {epoch+1} avg SupCon loss: {avg_loss:.4f}  lr: {optimizer.param_groups[0]['lr']:.2e}")

    # ===================== Phase 2: Linear Probe =====================
    # Freeze encoder, train FC head briefly so we can tune threshold
    print(f"\n--- Phase 2: Linear Probe ({LINEAR_PROBE_EPOCHS} epochs) ---")

    # Load regular (non-contrastive) datasets for CE training
    regular_datasets = []
    for patient_id in PATIENTS:
        if patient_id == exclude_patient:
            continue
        h5_path = Path("preprocessing") / "data" / patient_id / "train_dataset.h5"
        if h5_path.exists():
            regular_datasets.append(
                EEGDataset(str(h5_path), split=f"train ({patient_id})", augment=True)
            )

    combined_regular = ConcatDataset(regular_datasets)
    probe_loader = DataLoader(
        combined_regular,
        batch_size=SEQUENCE_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    # Freeze encoder, only train FC head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    all_labels = torch.cat([ds.labels for ds in regular_datasets])
    counts = torch.bincount(all_labels).float()
    class_weights = counts.sum() / (len(counts) * counts)
    ce_criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=0.05
    )
    probe_optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)

    for epoch in range(LINEAR_PROBE_EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(probe_loader, desc=f"Probe Epoch {epoch+1}/{LINEAR_PROBE_EPOCHS}")
        for x, labels in pbar:
            x, labels = x.to(device), labels.to(device)
            probe_optimizer.zero_grad()
            outputs = model(x)
            loss = ce_criterion(outputs, labels)
            loss.backward()
            probe_optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_loss = total_loss / len(probe_loader)
        print(f"  Probe Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Unfreeze all params before saving (so fine-tuning can train everything)
    for param in model.parameters():
        param.requires_grad = True

    # ===================== Save =====================
    if exclude_patient:
        save_dir = Path("model") / "contrastive" / f"lopo_{exclude_patient}"
    elif save_suffix:
        save_dir = Path("model") / "contrastive" / save_suffix
    else:
        save_dir = Path("model") / "contrastive"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "pretrained_encoder.pth"
    # Save only encoder (no projection head — it's discarded)
    torch.save(model.state_dict(), save_path)
    print(f"\nContrastive pretrained encoder saved to {save_path}")

    # ===================== Threshold Tuning =====================
    print("\nTuning threshold on combined cross-patient validation set...")
    val_datasets = []
    for patient_id in PATIENTS:
        if patient_id == exclude_patient:
            continue
        val_path = Path("preprocessing") / "data" / patient_id / "val_dataset.h5"
        if val_path.exists():
            val_datasets.append(EEGDataset(str(val_path), split=f"val ({patient_id})"))

    if val_datasets:
        combined_val = ConcatDataset(val_datasets)
        val_loader = DataLoader(
            combined_val,
            batch_size=SEQUENCE_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        print(f"Combined val set: {len(combined_val)} samples from {len(val_datasets)} patients")

        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for x, labels in tqdm(val_loader, desc="Val inference"):
                x, labels = x.to(device), labels.to(device)
                outputs = model(x)
                all_logits.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        logits_np = np.array(all_logits)
        labels_np = np.array(all_labels)

        # Temperature scaling
        print("\nCalibrating temperature scaling on validation set...")
        best_temperature = 1.0
        best_nll = float("inf")
        for t_candidate in np.arange(0.5, 5.05, 0.05):
            scaled_logits = torch.FloatTensor(logits_np) / t_candidate
            nll = nn.CrossEntropyLoss()(scaled_logits, torch.LongTensor(labels_np)).item()
            if nll < best_nll:
                best_nll = nll
                best_temperature = float(t_candidate)
        print(f"  Optimal temperature: {best_temperature:.2f} (NLL: {best_nll:.4f})")

        scaled_logits = torch.FloatTensor(logits_np) / best_temperature
        probs_np = torch.softmax(scaled_logits, dim=1)[:, 1].numpy()

        if len(np.unique(labels_np)) > 1:
            fpr, tpr, thresholds = roc_curve(labels_np, probs_np)
            j_scores = tpr - fpr
            optimal_threshold = float(thresholds[np.argmax(j_scores)])
        else:
            optimal_threshold = 0.5

        optimized_preds = (probs_np >= optimal_threshold).astype(int)
        best_window, best_count = optimize_running_average(optimized_preds, labels_np)

        pretrained_config = {
            "optimal_threshold": optimal_threshold,
            "smoothing_window": best_window,
            "smoothing_count": best_count,
            "temperature": best_temperature,
            "task_mode": TASK_MODE,
        }
        config_path = save_dir / "pretrained_config.json"
        with open(config_path, "w") as f:
            json.dump(pretrained_config, f, indent=2)

        print(f"  Pretrained threshold: {optimal_threshold:.4f}")
        print(f"  Smoothing: window={best_window}, count={best_count}")
        print(f"  Config saved to {config_path}")
    else:
        print("  No val datasets found, skipping threshold tuning")


def main():
    parser = argparse.ArgumentParser(description="EEG Conv-Transformer Training")
    parser.add_argument(
        "--pretrain", action="store_true",
        help="Run cross-patient pretraining instead of per-patient fine-tuning"
    )
    parser.add_argument(
        "--lopo", type=str, default=None,
        help="Leave-One-Patient-Out: pretrain excluding this patient, then fine-tune on it. "
             "Example: --lopo chb01"
    )
    parser.add_argument(
        "--ensemble", type=int, default=None,
        help="Train an ensemble of N pretrained encoders with different seeds, "
             "then fine-tune each on all patients. Example: --ensemble 5"
    )
    args = parser.parse_args()

    if args.ensemble:
        n_seeds = args.ensemble
        print("=" * 60)
        print(f"ENSEMBLE TRAINING: {n_seeds} seeds")
        print("=" * 60)

        # Step 1: Pretrain N encoders with different seeds
        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx * 100  # spread seeds apart
            suffix = f"seed{seed_idx}"
            print(f"\n{'='*60}")
            print(f"ENSEMBLE: Pretraining seed {seed_idx+1}/{n_seeds} (seed={seed})")
            print(f"{'='*60}")
            pretrain(seed=seed, save_suffix=suffix)

        # Step 2: Fine-tune each encoder on all patients
        n_patients = len(PATIENTS)
        if PATIENT_INDEX is None:
            patients_to_process = list(range(n_patients))
        else:
            patients_to_process = [PATIENT_INDEX]

        for seed_idx in range(n_seeds):
            suffix = f"seed{seed_idx}"
            encoder_path = Path("model") / "contrastive" / suffix / "pretrained_encoder.pth"
            print(f"\n{'='*60}")
            print(f"ENSEMBLE: Fine-tuning with {suffix}")
            print(f"{'='*60}")

            for current_idx in patients_to_process:
                patient_config = get_patient_config(current_idx)
                try:
                    EEGTrainer(
                        patient_config,
                        pretrained_path=encoder_path,
                        model_subdir=suffix,
                    ).train()
                except Exception as e:
                    print(f"Error {patient_config['patient_id']} ({suffix}): {e}")
                    import traceback
                    traceback.print_exc()
        return

    if args.lopo:
        # LOPO mode: pretrain excluding target patient, then fine-tune on it
        target_patient = args.lopo
        if target_patient not in PATIENTS:
            print(f"Error: {target_patient} not in PATIENTS list")
            return

        print("=" * 60)
        print(f"LEAVE-ONE-PATIENT-OUT: {target_patient}")
        print("=" * 60)

        # Step 1: Pretrain excluding target patient
        pretrain(exclude_patient=target_patient)

        # Step 2: Fine-tune on target patient using LOPO encoder
        lopo_encoder_path = Path("model") / "contrastive" / f"lopo_{target_patient}" / "pretrained_encoder.pth"
        patient_idx = PATIENTS.index(target_patient)
        patient_config = get_patient_config(patient_idx)
        try:
            EEGTrainer(patient_config, pretrained_path=lopo_encoder_path).train()
        except Exception as e:
            print(f"Error {target_patient}: {e}")
            import traceback
            traceback.print_exc()
        return

    if args.pretrain:
        pretrain()
        return

    n_patients = len(PATIENTS)
    if PATIENT_INDEX is None:
        patients_to_process = list(range(n_patients))
    else:
        patients_to_process = [PATIENT_INDEX]

    for current_idx in patients_to_process:
        patient_config = get_patient_config(current_idx)
        try:
            EEGTrainer(patient_config).train()
        except Exception as e:
            print(f"Error {patient_config['patient_id']}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
