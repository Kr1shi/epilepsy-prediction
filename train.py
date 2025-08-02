import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import h5py
import numpy as np
import json
import os
from pathlib import Path
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
            self.spectrograms = torch.FloatTensor(f['spectrograms'][:])
            self.labels = torch.LongTensor(f['labels'][:])
            self.patient_ids = [pid.decode('utf-8') for pid in f['patient_ids'][:]]
            
            # Load metadata
            if 'metadata' in f:
                self.metadata = dict(f['metadata'].attrs)
            else:
                self.metadata = {}
        
        print(f"Loaded {self.split} dataset: {len(self.spectrograms)} samples")
        print(f"  - Spectrogram shape: {self.spectrograms.shape}")
        print(f"  - Class distribution: {torch.bincount(self.labels)}")
        
    def __len__(self):
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        """
        Returns:
            spectrogram: (n_channels, n_frequencies, n_time_windows)
            label: int (0=interictal, 1=preictal)
        """
        return self.spectrograms[idx], self.labels[idx]

class Modified_ResNet18(nn.Module):
    """ResNet18 modified for EEG spectrograms with 18 input channels"""
    
    def __init__(self, num_input_channels=18, num_classes=2):
        super(Modified_ResNet18, self).__init__()
        
        # Load ResNet18 architecture (without pretrained weights)
        self.resnet = models.resnet18(pretrained=False)
        
        # Modify first convolutional layer for 18-channel input
        self.resnet.conv1 = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Modify final layer for binary classification
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
        # Initialize new layers with proper weights
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_uniform_(self.resnet.fc.weight)
        nn.init.zeros_(self.resnet.fc.bias)
        
    def forward(self, x):
        return self.resnet(x)

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
    
    def __init__(self):
        # Setup directories
        self.model_dir = Path("model")
        self.model_dir.mkdir(exist_ok=True)
        
        # Device setup with MPS support for Apple Silicon
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        
        # Load datasets
        self.train_loader = self._create_dataloader('train')
        self.val_loader = self._create_dataloader('val')
        
        # Initialize model
        self.model = Modified_ResNet18(num_input_channels=18, num_classes=2)
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # Metrics tracking
        self.train_metrics_history = []
        self.val_metrics_history = []
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
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
        h5_file = f"preprocessing/data/{split}_dataset.h5"
        
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"Dataset file not found: {h5_file}")
        
        dataset = EEGDataset(h5_file, split=split)
        
        # Use shuffle for training, not for validation
        shuffle = (split == 'train')
        
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
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
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
            predictions = torch.argmax(outputs, dim=1)
            
            train_metrics.update(predictions, labels, probabilities)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
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
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': {
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY
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
                'epochs': TRAINING_EPOCHS,
                'batch_size': BATCH_SIZE,
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
        fig.suptitle('EEG Seizure Prediction - Training Progress', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.train_metrics_history) + 1)
        
        # Plot each metric
        metrics_to_plot = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        
        for idx, metric in enumerate(metrics_to_plot):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            train_values = [m[metric] for m in self.train_metrics_history]
            val_values = [m[metric] for m in self.val_metrics_history]
            
            ax.plot(epochs, train_values, 'o-', label='Train', linewidth=2, markersize=4)
            ax.plot(epochs, val_values, 's-', label='Validation', linewidth=2, markersize=4)
            
            ax.set_title(f'{metric.upper().replace("_", " ")}', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add best value annotation
            if metric == 'loss':
                best_val_idx = np.argmin(val_values)
                best_val = val_values[best_val_idx]
            else:
                best_val_idx = np.argmax(val_values)
                best_val = val_values[best_val_idx]
            
            ax.annotate(f'Best: {best_val:.3f}', 
                       xy=(best_val_idx + 1, best_val), 
                       xytext=(10, 10), 
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
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
        print("STARTING EEG SEIZURE PREDICTION TRAINING")
        print("="*60)
        print(f"Training for {TRAINING_EPOCHS} epochs")
        print(f"Batch size: {BATCH_SIZE}")
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
            
            # Train and validate
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            
            # Store metrics
            self.train_metrics_history.append(train_metrics)
            self.val_metrics_history.append(val_metrics)
            
            # Update learning rate scheduler
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics['loss'])
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Log learning rate changes
            if new_lr != old_lr:
                print(f"📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
            
            # Save model checkpoint
            self.save_model(epoch, train_metrics, val_metrics)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{TRAINING_EPOCHS} Complete ({epoch_time:.1f}s) [LR: {current_lr:.6f}]")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc_roc']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
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
    trainer = EEGCNNTrainer()
    trainer.train()

if __name__ == "__main__":
    main()