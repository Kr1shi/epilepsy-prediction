import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import h5py
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from data_segmentation_helpers.config import *

class EEGDataset(Dataset):
    """Custom dataset for loading EEG spectrograms from HDF5 files"""
    
    def __init__(self, h5_file_path, split='train'):
        """
        Args:
            h5_file_path (str): Path to HDF5 file
            split (str): 'train', 'val', or 'test'
        """
        self.h5_file_path = h5_file_path
        self.split = split
        
        # Load all data into memory for fastest training
        with h5py.File(h5_file_path, 'r') as f:
            # Data is already normalized during preprocessing (z-score normalization)
            self.spectrograms = torch.FloatTensor(f['spectrograms'][:])
            self.labels = torch.LongTensor(f['labels'][:])
            self.patient_ids = [pid.decode('utf-8') for pid in f['patient_ids'][:]]

            # Load metadata
            if 'metadata' in f:
                self.metadata = dict(f['metadata'].attrs)
                # Check if normalization info is available
                if 'normalization' in self.metadata:
                    norm_info = json.loads(self.metadata['normalization'])
                    print(f"  - Normalization: {norm_info['method']}, "
                          f"mean={norm_info['global_mean']:.4f}, std={norm_info['global_std']:.4f}")
            else:
                self.metadata = {}

        print(f"Loaded {self.split} dataset: {len(self.spectrograms)} samples")
        print(f"  - Spectrogram shape: {self.spectrograms.shape}")
        print(f"  - Value range: [{torch.min(self.spectrograms):.4f}, {torch.max(self.spectrograms):.4f}]")
        print(f"  - Class distribution: {torch.bincount(self.labels)}")
        
    def __len__(self):
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        """
        Returns:
            spectrogram: (sequence_length, n_channels, n_frequencies, n_time_windows)
            label: int (0=interictal, 1=preictal/ictal depending on task_mode)
        """
        return self.spectrograms[idx], self.labels[idx]

class EEG_CNN(nn.Module):
    """Deep VGG-16 style CNN designed for EEG spectrograms

    Architecture: 5 blocks, 16 convolutional layers with progressive widening
    Adapted for 5-second segments (50×9 spectrograms)
    - Input: (channels, 50, 9)
    - Block1: → (128, 25, 4)    [3 conv, 2x downsampling]
    - Block2: → (256, 12, 2)    [3 conv, 2x downsampling]
    - Block3: → (512, 6, 1)     [3 conv, 2x downsampling]
    - Block4: → (512, 6, 1)     [4 conv, no downsampling]
    - Block5: → (512, 6, 1)     [3 conv, no downsampling]
    - GAP:    → (512, 1, 1)
    - Output: 512 features

    Total: 16 conv layers, ~9.5M parameters (vs previous 6 layers, ~420K parameters)
    Provides significantly more capacity for learning complex EEG patterns
    """

    def __init__(self, in_channels=18):
        super(EEG_CNN, self).__init__()

        # Block 1: Low-level frequency-time pattern extraction
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 50×59 → 25×29
        )

        # Block 2: Mid-level feature learning
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 25×29 → 12×14
        )

        # Block 3: High-level feature learning
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 12×14 → 6×7
        )

        # Block 4: Deep high-level features (no pooling - maintain spatial dimensions)
        self.block4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            # No pooling - maintain 6×1 spatial dimensions for small input size
        )

        # Block 5: Very deep feature refinement
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            # No pooling - maintain 3×3 spatial dimensions
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # 3×3 → 1×1

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width) = (batch, 18, 50, 9)
        Returns:
            features: (batch, 512)
        """
        x = self.block1(x)  # (batch, 128, 25, 4)
        x = self.block2(x)  # (batch, 256, 12, 2)
        x = self.block3(x)  # (batch, 512, 6, 1)
        x = self.block4(x)  # (batch, 512, 6, 1)
        x = self.block5(x)  # (batch, 512, 6, 1)
        x = self.gap(x)     # (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 512)
        return x

class CNN_LSTM_Hybrid(nn.Module):
    """Deep CNN-BiLSTM Hybrid model for seizure prediction/detection

    Architecture:
    1. Deep EEG-CNN (VGG-16 style) as feature extractor (per segment)
       - 5 blocks, 16 convolutional layers
       - Channel progression: 128 → 256 → 512 → 512 → 512
       - Outputs 512-dimensional feature vector per segment
       - ~9.5M parameters for rich feature learning
    2. Bidirectional LSTM for temporal modeling across sequence
       - 3 layers, 512 hidden units per direction (1024 total)
       - Processes sequence in both forward and backward directions
       - 50% dropout for regularization
       - ~13M parameters
    3. Attention mechanism for adaptive temporal pooling
    4. Fully connected classification head

    The deep CNN provides significantly more capacity (~23x more parameters)
    for learning complex spectro-temporal patterns in EEG data. The bidirectional
    LSTM captures both past and future temporal context, improving prediction
    accuracy. The attention mechanism learns which timesteps in the sequence are
    most important for classification, replacing simple "last hidden state" pooling
    with a weighted combination of all timesteps.

    Total: ~22.5M parameters (9.5M CNN + 13M BiLSTM)

    Supports two task modes:
    - Prediction: Classify preictal vs interictal (predict before seizure)
    - Detection: Classify ictal vs interictal (detect during seizure)
    """

    def __init__(self,
                 num_input_channels=18,
                 num_classes=2,
                 sequence_length=SEQUENCE_LENGTH,
                 cnn_feature_dim=512,
                 lstm_hidden_dim=LSTM_HIDDEN_DIM,
                 lstm_num_layers=LSTM_NUM_LAYERS,
                 dropout=LSTM_DROPOUT):
        super(CNN_LSTM_Hybrid, self).__init__()

        self.sequence_length = sequence_length
        self.cnn_feature_dim = cnn_feature_dim

        # CNN Feature Extractor (Custom EEG-CNN)
        self.feature_extractor = self._build_eeg_cnn_backbone(num_input_channels)

        # Bidirectional LSTM for temporal modeling (processes sequence in both directions)
        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism - learns which timesteps are most important
        # Input: lstm_hidden_dim * 2 due to bidirectional LSTM (forward + backward)
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )

        # Classification head
        # Input: lstm_hidden_dim * 2 due to bidirectional LSTM (forward + backward)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # Initialize classification head
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _build_eeg_cnn_backbone(self, num_input_channels):
        """Build custom EEG-CNN backbone"""
        return EEG_CNN(in_channels=num_input_channels)

    def forward(self, x):
        """
        Args:
            x: (batch, sequence_length, channels, freq, time)
        Returns:
            output: (batch, num_classes)
        """
        batch_size, seq_len, c, h, w = x.shape

        # Process each segment through CNN
        # Reshape: (batch * seq_len, channels, freq, time)
        x = x.view(batch_size * seq_len, c, h, w)

        # Extract CNN features: (batch * seq_len, cnn_feature_dim)
        features = self.feature_extractor(x)

        # Reshape back to sequence: (batch, seq_len, cnn_feature_dim)
        features = features.view(batch_size, seq_len, -1)

        # LSTM: (batch, seq_len, lstm_hidden_dim)
        lstm_out, _ = self.lstm(features)

        # Attention pooling - learn which timesteps matter most
        attn_weights = self.attention(lstm_out)        # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # softmax over timesteps
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, lstm_hidden_dim)

        # Classification: (batch, num_classes)
        output = self.fc(context)

        return output

class MetricsTracker:
    """Track and compute comprehensive metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
    
    def update(self, predictions, labels, probabilities):
        """
        Args:
            predictions: predicted class labels
            labels: true class labels  
            probabilities: predicted probabilities for positive class
        """
        self.predictions.extend(predictions.detach().cpu().numpy())
        self.true_labels.extend(labels.detach().cpu().numpy())
        self.probabilities.extend(probabilities.detach().cpu().numpy())
    
    def compute_metrics(self):
        """Compute all metrics"""
        pred_np = np.array(self.predictions)
        true_np = np.array(self.true_labels)
        prob_np = np.array(self.probabilities)
        
        metrics = {
            'accuracy': accuracy_score(true_np, pred_np),
            'precision': precision_score(true_np, pred_np, average='binary', zero_division=0),
            'recall': recall_score(true_np, pred_np, average='binary', zero_division=0),
            'f1': f1_score(true_np, pred_np, average='binary', zero_division=0),
            'auc_roc': roc_auc_score(true_np, prob_np) if len(np.unique(true_np)) > 1 else 0.0
        }
        
        return metrics

class EEGCNNTrainer:
    """Main training class for EEG seizure prediction CNN"""

    def __init__(self, fold_config: Dict = None):
        # Use fold_config if provided, otherwise use global config
        if fold_config is not None:
            dataset_prefix = fold_config['output_prefix']
            self.fold_id = fold_config['fold_id']
        else:
            dataset_prefix = OUTPUT_PREFIX
            self.fold_id = LOOCV_FOLD_ID

        # Setup directories (fold-specific)
        self.model_dir = Path("model") / dataset_prefix
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_prefix = dataset_prefix
        self.dataset_dir = Path("preprocessing") / "data" / self.dataset_prefix
        print(f"Using dataset prefix: {self.dataset_prefix}")
        print(f"Loading datasets from: {self.dataset_dir}")
        print(f"Saving models to: {self.model_dir}")

        # Device setup with MPS support for Apple Silicon
        self.device = self._get_device()
        print(f"Using device: {self.device}")

        # Load datasets
        self.train_loader = self._create_dataloader('train')
        # LOOCV mode: no validation set
        self.val_loader = None
        print(f"LOOCV Mode (Fold {self.fold_id}): No validation set, using training-only evaluation")
        
        # Initialize CNN-LSTM model with deep EEG-CNN backbone
        self.model = CNN_LSTM_Hybrid(
            num_input_channels=18,
            num_classes=2,
            sequence_length=SEQUENCE_LENGTH,
            cnn_feature_dim=512,  # Deep EEG-CNN outputs 512 features (16 conv layers)
            lstm_hidden_dim=LSTM_HIDDEN_DIM,
            lstm_num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT
        )
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Learning rate scheduler - StepLR reduces LR every N epochs
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=5,      # Reduce LR every 10 epochs
            gamma=0.5          # Multiply LR by 0.5
        )
        
        # Metrics tracking
        self.train_metrics_history = []
        self.val_metrics_history = []
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")

    @property
    def positive_label(self):
        """Get positive class label based on task mode"""
        return 'preictal' if TASK_MODE == 'prediction' else 'ictal'

    def _get_device_name(self):
        """Get descriptive device name"""
        if self.device.type == 'cuda':
            return torch.cuda.get_device_name()
        elif self.device.type == 'mps':
            return "Apple Silicon GPU (MPS)"
        else:
            return "CPU"
    
    def _get_device(self):
        """Detect best available device with preference: CUDA > MPS > CPU"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🚀 CUDA detected: {torch.cuda.get_device_name()}")
            return device
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"🍎 MPS detected: Using Apple Silicon GPU acceleration")
            return device
        else:
            device = torch.device("cpu")
            print(f"💻 Using CPU (consider upgrading to GPU for faster training)")
            return device
        """Detect best available device with preference: CUDA > MPS > CPU"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🚀 CUDA detected: {torch.cuda.get_device_name()}")
            return device
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"🍎 MPS detected: Using Apple Silicon GPU acceleration")
            return device
        else:
            device = torch.device("cpu")
            print(f"💻 Using CPU (consider upgrading to GPU for faster training)")
            return device
        
    def _create_dataloader(self, split):
        """Create dataloader for given split"""
        h5_file = self.dataset_dir / f"{split}_dataset.h5"
        
        if not h5_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {h5_file}")
        
        dataset = EEGDataset(str(h5_file), split=split)
        
        # Use shuffle for training, not for validation
        shuffle = (split == 'train')
        
        dataloader = DataLoader(
            dataset,
            batch_size=SEQUENCE_BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True if self.device.type == 'cuda' else False  # Only CUDA supports pin_memory
        )
        
        return dataloader
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        train_metrics = MetricsTracker()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Training Epoch {epoch+1}/{TRAINING_EPOCHS}')
        
        for batch_idx, (spectrograms, labels) in enumerate(pbar):
            spectrograms, labels = spectrograms.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(spectrograms)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()

            self.optimizer.step()
            # Note: scheduler.step() is now called per-epoch, not per-batch

            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
            predictions = torch.argmax(outputs, dim=1)
            
            train_metrics.update(predictions, labels, probabilities)
            
            # Calculate gradient norm for monitoring
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            # Update progress bar with loss and gradient norm
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'grad_norm': f'{total_norm:.3f}',
                'lr': f'{current_lr:.6f}'
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        metrics = train_metrics.compute_metrics()
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        val_metrics = MetricsTracker()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Validation Epoch {epoch+1}/{TRAINING_EPOCHS}')
            
            for spectrograms, labels in pbar:
                spectrograms, labels = spectrograms.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(spectrograms)
                loss = self.criterion(outputs, labels)
                
                # Track metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                predictions = torch.argmax(outputs, dim=1)
                
                val_metrics.update(predictions, labels, probabilities)
                
                # Update progress bar
                pbar.set_postfix({'val_loss': loss.item()})
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        metrics = val_metrics.compute_metrics()
        metrics['loss'] = avg_loss
        
        return metrics
    
    def save_model(self, epoch, train_metrics, val_metrics):
        """Save model checkpoint (fold-specific)"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': {
                'task_mode': TASK_MODE,
                'positive_class': self.positive_label,
                'negative_class': 'interictal',
                'batch_size': SEQUENCE_BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
                'loocv_fold': LOOCV_FOLD_ID,
                'loocv_patient': SINGLE_PATIENT_ID
            }
        }

        model_path = self.model_dir / f"epoch_{epoch+1:03d}.pth"
        torch.save(checkpoint, model_path)
        
    def save_metrics(self):
        """Save training metrics to JSON"""
        metrics_data = {
            'train_metrics': self.train_metrics_history,
            'val_metrics': self.val_metrics_history,
            'config': {
                'task_mode': TASK_MODE,
                'positive_class': self.positive_label,
                'negative_class': 'interictal',
                'epochs': TRAINING_EPOCHS,
                'batch_size': SEQUENCE_BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY
            },
            'training_info': {
                'device': str(self.device),
                'device_name': self._get_device_name(),
                'model_parameters': sum(p.numel() for p in self.model.parameters()),
                'training_time': datetime.now().isoformat()
            }
        }
        
        metrics_path = self.model_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def plot_training_curves(self):
        """Create and save training curve visualizations"""
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # LOOCV mode: include fold information in title
        title = f'EEG Seizure {TASK_MODE.capitalize()} - Fold {LOOCV_FOLD_ID} Training Progress ({self.positive_label} vs interictal)'
        fig.suptitle(title, fontsize=16, fontweight='bold')

        epochs = range(1, len(self.train_metrics_history) + 1)

        # Plot each metric
        metrics_to_plot = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc']

        for idx, metric in enumerate(metrics_to_plot):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            train_values = [m[metric] for m in self.train_metrics_history]

            ax.plot(epochs, train_values, 'o-', label='Train', linewidth=2, markersize=4)

            # Only plot validation if available
            if self.val_metrics_history:
                val_values = [m[metric] for m in self.val_metrics_history]
                ax.plot(epochs, val_values, 's-', label='Validation', linewidth=2, markersize=4)
            
            ax.set_title(f'{metric.upper().replace("_", " ")}', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add best value annotation (use validation if available, otherwise training)
            if self.val_metrics_history:
                if metric == 'loss':
                    best_val_idx = np.argmin(val_values)
                    best_val = val_values[best_val_idx]
                else:
                    best_val_idx = np.argmax(val_values)
                    best_val = val_values[best_val_idx]

                ax.annotate(f'Best Val: {best_val:.3f}',
                           xy=(best_val_idx + 1, best_val),
                           xytext=(10, 10),
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            else:
                # LOOCV mode: annotate best training value
                if metric == 'loss':
                    best_train_idx = np.argmin(train_values)
                    best_train = train_values[best_train_idx]
                else:
                    best_train_idx = np.argmax(train_values)
                    best_train = train_values[best_train_idx]

                ax.annotate(f'Best Train: {best_train:.3f}',
                           xy=(best_train_idx + 1, best_train),
                           xytext=(10, 10),
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.model_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {plot_path}")
    
    def train(self):
        """Main training loop"""
        print("="*60)
        print(f"STARTING EEG SEIZURE {TASK_MODE.upper()} TRAINING")
        print("="*60)
        print(f"Task mode: {TASK_MODE.upper()} ({self.positive_label} vs interictal)")
        print(f"LOOCV Mode: Fold {LOOCV_FOLD_ID}/{LOOCV_TOTAL_SEIZURES-1} (Test seizure: {LOOCV_FOLD_ID})")
        print(f"Note: No validation set in LOOCV mode")
        print(f"Training for {TRAINING_EPOCHS} epochs")
        print(f"Batch size: {SEQUENCE_BATCH_SIZE}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Device: {self.device} ({self._get_device_name()})")
        if self.device.type == 'mps':
            print("🍎 Using Apple Silicon GPU - expect ~3-5x speedup vs CPU!")
        elif self.device.type == 'cuda':
            print("🚀 Using CUDA GPU - maximum performance!")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(TRAINING_EPOCHS):
            epoch_start_time = time.time()
            
            # Train (no validation in LOOCV mode)
            train_metrics = self.train_epoch(epoch)
            val_metrics = None

            # Store metrics
            self.train_metrics_history.append(train_metrics)

            # Update learning rate (StepLR updates per-epoch)
            self.scheduler.step()

            # Save model checkpoint
            self.save_model(epoch, train_metrics, val_metrics)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"\nEpoch {epoch+1}/{TRAINING_EPOCHS} Complete ({epoch_time:.1f}s) [LR: {current_lr:.6f}]")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
                  f"Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc_roc']:.4f}")
            if val_metrics is not None:
                print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                      f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
                      f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc_roc']:.4f}")
            print("-" * 60)
        
        # Training complete
        total_time = time.time() - start_time
        
        # Save final metrics and plots
        self.save_metrics()
        self.plot_training_curves()
        
        print("="*60)
        print("TRAINING COMPLETED!")
        print(f"Total training time: {total_time/60:.1f} minutes")
        print(f"Models saved in: {self.model_dir}")
        print("="*60)

def main():
    """Main execution function"""
    # Determine which folds to process
    if LOOCV_FOLD_ID is None:
        folds_to_process = list(range(LOOCV_TOTAL_SEIZURES))
        print("="*60)
        print("BATCH PROCESSING: ALL FOLDS")
        print(f"Processing {len(folds_to_process)} folds for patient {SINGLE_PATIENT_ID}")
        print("="*60)
    else:
        folds_to_process = [LOOCV_FOLD_ID]
        print("="*60)
        print("SINGLE FOLD PROCESSING")
        print(f"Processing fold {LOOCV_FOLD_ID} for patient {SINGLE_PATIENT_ID}")
        print("="*60)

    # Process each fold
    for current_fold in folds_to_process:
        fold_config = get_fold_config(current_fold)

        print(f"\n{'='*60}")
        print(f"TRAINING FOLD {current_fold}/{LOOCV_TOTAL_SEIZURES-1}")
        print(f"{'='*60}")

        try:
            # Run training with fold-specific config
            trainer = EEGCNNTrainer(fold_config=fold_config)
            trainer.train()

            print(f"✅ Fold {current_fold} training completed successfully!")
        except Exception as e:
            print(f"❌ Error training fold {current_fold}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    if LOOCV_FOLD_ID is None:
        print("\n" + "="*60)
        print(f"✅ BATCH TRAINING COMPLETED!")
        print(f"✅ Trained models for {len(folds_to_process)} folds for patient {SINGLE_PATIENT_ID}")
        print("="*60)

if __name__ == "__main__":
    main()
