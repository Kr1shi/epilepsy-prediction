# Epilepsy Seizure Prediction

A deep learning system for predicting epileptic seizures using EEG signals from the CHB-MIT Scalp EEG Database. The system classifies EEG sequences as **preictal** (before a seizure) or **interictal** (seizure-free) using a **CNN-LSTM hybrid architecture** that combines spatial feature extraction with temporal modeling across time.

## Overview

This project implements a complete pipeline from raw EEG data to trained seizure prediction model:

1. **Data Segmentation**: Extract labeled **sequences** of consecutive EEG segments from raw recordings
2. **Data Preprocessing**: Transform EEG sequences into spectrograms with patient-level splits
3. **Model Training**: Train a CNN-LSTM hybrid model for binary classification
4. **Evaluation**: Test the trained model and generate performance metrics

## Architecture Highlights

- **CNN-LSTM Hybrid**: ResNet18 backbone extracts spatial features from each segment, LSTM models temporal patterns across the sequence
- **Temporal Context**: Uses sequences of 10 consecutive 30-second segments (5 minutes of EEG activity)
- **Patient-Level Splits**: Train/val/test split by patient (not by segments) for better generalization
- **Sequence Length**: `SEQUENCE_LENGTH` segments per sequence with sliding window stride

## Dataset

**Source**: CHB-MIT Scalp EEG Database from PhysioNet

**Patients**: 23 patients (chb01-chb24, excluding chb12)

**Channels**: 18 bipolar EEG channels from the 10-20 system

**Format**: European Data Format (EDF) files

**Location**: `physionet.org/files/chbmit/1.0.0/`

## Prerequisites

### Python Version
- Python 3.x

### Required Packages
```bash
pip install mne numpy h5py tqdm scikit-learn
```

Additional packages for training and visualization:
```bash
pip install scipy matplotlib
```

## Pipeline Steps

### Step 1: Data Segmentation (Sequences)

Extract **sequences of consecutive EEG segments** and label them as preictal or interictal.

```bash
python data_segmentation.py
```

**What it does**:
- Parses patient summary files to identify seizure times
- Extracts **sequences** of `SEQUENCE_LENGTH` consecutive segments using a sliding window
- Labels sequences based on the **last segment** in the sequence
- Extracts **preictal sequences** from the time window (`PREICTAL_WINDOW`) before each seizure
- Extracts **interictal sequences** from seizure-free recordings
- Saves sequence metadata to `all_patients_sequences.json`

**Configuration**: See [data_segmentation_helpers/config.py](data_segmentation_helpers/config.py)

Key parameters:
- `SEGMENT_DURATION`: Duration of each individual segment within a sequence (30 seconds)
- `SEQUENCE_LENGTH`: Number of consecutive segments per sequence (10 = 5 minutes total)
- `SEQUENCE_STRIDE`: Sliding window stride in segments (5 = 50% overlap)
- `PREICTAL_WINDOW`: Time window before seizure to extract preictal sequences
- `TARGET_CHANNELS`: List of 18 bipolar EEG channel pairs to use
- `ESTIMATED_FILE_DURATION`: Assumed duration of each recording file

### Step 2: Data Preprocessing (Sequences)

Transform raw EEG sequences into spectrograms using signal processing techniques.

```bash
python data_preprocessing.py
```

**What it does**:
- Loads sequences from `all_patients_sequences.json`
- Applies **patient-level train/val/test split** (70/15/15) for better generalization
- For each sequence, processes all segments through the pipeline:
  1. Artifact removal using Median Absolute Deviation (threshold: `ARTIFACT_THRESHOLD_STD`)
  2. Bandpass filtering (`LOW_FREQ_HZ` to `HIGH_FREQ_HZ`)
  3. Notch filtering (`NOTCH_FREQ_HZ` to remove electrical line noise)
  4. Robust normalization per channel
  5. STFT to generate frequency-time spectrograms (`STFT_NPERSEG`, `STFT_NOVERLAP`)
  6. Power conversion (magnitude squared, log-transformed if `APPLY_LOG_TRANSFORM`)
- Stacks segments into sequences: shape `(sequence_length, 18, freq_bins, time_bins)`
- Saves HDF5 datasets to `preprocessing/data/`

**Outputs**:
- `train_dataset.h5` - sequences from training patients
- `val_dataset.h5` - sequences from validation patients
- `test_dataset.h5` - sequences from held-out test patients

**Key Difference from CNN-only**: Uses patient-level splits (not random) to ensure model generalizes to unseen patients.

**Features**:
- Checkpointing: Resume from `preprocessing/checkpoints/progress.json`
- Incremental saving for robustness
- Detailed logging in `preprocessing/logs/preprocessing.log`

**Configuration parameters** ([data_segmentation_helpers/config.py](data_segmentation_helpers/config.py)):
- `LOW_FREQ_HZ`: High-pass filter cutoff
- `HIGH_FREQ_HZ`: Low-pass filter cutoff
- `NOTCH_FREQ_HZ`: Notch filter frequency (removes line noise)
- `ARTIFACT_THRESHOLD_STD`: Standard deviations for artifact detection
- `STFT_NPERSEG`: STFT window length
- `STFT_NOVERLAP`: STFT window overlap
- `LOG_TRANSFORM_EPSILON`: Small value to avoid log(0)
- `APPLY_LOG_TRANSFORM`: Whether to apply log transform to spectrograms
- `SKIP_CHANNEL_VALIDATION`: Skip initial channel validation for faster startup

### Step 2.5 (Optional): Data Visualization

Visualize sample spectrograms before running full training pipeline.

```bash
python data_visualization.py
```

**What it does**:
- Randomly selects preictal and interictal segments
- Applies the same preprocessing pipeline
- Generates spectrogram visualizations
- Saves images to `preprocessing/visualizations/`

This helps verify signal processing quality and configuration correctness.

### Step 3: Model Training (CNN-LSTM)

Train a CNN-LSTM hybrid model on the preprocessed sequence spectrograms.

```bash
python train.py
```

**What it does**:
- Loads HDF5 sequence datasets from `preprocessing/data/`
- Initializes **CNN-LSTM Hybrid** model:
  - **CNN Feature Extractor**: ResNet18 backbone (modified for 18-channel input) extracts spatial features from each segment
  - **LSTM**: Models temporal dependencies across the sequence of segments
  - **Classification Head**: Fully connected layers for binary classification
- Trains using Adam optimizer with ReduceLROnPlateau scheduling
- Validates after each epoch
- Saves checkpoints to `model/epoch_XXX.pth`
- Logs metrics to `model/training_metrics.json`
- Generates training curves in `model/training_curves.png`

**Training Configuration** ([data_segmentation_helpers/config.py](data_segmentation_helpers/config.py)):
- `TRAINING_EPOCHS`: Number of training epochs
- `SEQUENCE_BATCH_SIZE`: Batch size for sequences (reduced from individual segments)
- `LEARNING_RATE`: Initial learning rate for Adam optimizer
- `WEIGHT_DECAY`: L2 regularization coefficient
- `NUM_WORKERS`: Number of DataLoader worker processes

**LSTM Configuration**:
- `LSTM_HIDDEN_DIM`: LSTM hidden state dimension (256)
- `LSTM_NUM_LAYERS`: Number of stacked LSTM layers (2)
- `LSTM_DROPOUT`: Dropout between LSTM layers (0.3)

**Device Support**:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU fallback

**Model Architecture**:
```
Input: (batch, 10, 18, freq, time) - sequence of 10 spectrograms
   ↓
CNN Feature Extractor (ResNet18 per segment)
   ↓
Features: (batch, 10, 512) - 512 features per segment
   ↓
LSTM (2 layers, hidden_dim=256)
   ↓
Last Hidden State: (batch, 256)
   ↓
FC Layers: 256 → 128 → 2
   ↓
Output: (batch, 2) - preictal/interictal logits
```
- Parameters: ~14 million (ResNet18 + LSTM + FC layers)

### Step 4: Model Evaluation

Evaluate the trained model on the held-out test set.

```bash
python evaluate_test.py
```

**What it does**:
- Loads trained model weights from `model/epoch_030.pth`
- Runs inference on test dataset
- Computes classification metrics:
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix
  - ROC curve and AUC score
- Saves results to `model/test_results.json`
- Generates visualization plots

## Single-Patient Experiment Mode

To run the entire pipeline on one subject, enable the single-patient switches in [data_segmentation_helpers/config.py](data_segmentation_helpers/config.py):

- `SINGLE_PATIENT_MODE = True`
- `SINGLE_PATIENT_ID = "chb06"` (recommended starting point: 10 documented seizures and consistent channel coverage)
- `SINGLE_PATIENT_SEIZURE_SPLITS`: map seizure indices to `train` / `val` / `test` (default: first 6 seizures for training, 2 for validation, 2 for testing)
- `SINGLE_PATIENT_INTERICTAL_SPLIT`: ratios for distributing interictal sequences (defaults to 60/20/20)

With these flags enabled:
- `python data_segmentation.py` writes `chb06_sequences_detection.json` (prefix follows `OUTPUT_PREFIX`) and tags each sequence with its split.
- Each split is then auto-balanced (equal positives and interictals) by downsampling the majority class before saving the JSON.
- `python data_preprocessing.py` reads that JSON, keeps the existing split assignments, and saves HDF5 datasets to `preprocessing/data/chb06/{train,val,test}_dataset.h5` with matching checkpoints/logs under `preprocessing/checkpoints/chb06/` and `preprocessing/logs/chb06/`.
- `python train.py` and `python evaluate_test.py` automatically pick up the same prefix so they train/evaluate purely on the single-patient datasets.

Notes:
- Any positive sequence whose seizure index is not listed in `SINGLE_PATIENT_SEIZURE_SPLITS` is dropped, so adjust the mapping if you need additional seizures.
- Interictal sequences are deterministically shuffled using `SINGLE_PATIENT_RANDOM_SEED` before applying the ratio split.

## Configuration

All parameters are centralized in [data_segmentation_helpers/config.py](data_segmentation_helpers/config.py).

### Sequence Configuration (CNN-LSTM)
- `SEQUENCE_LENGTH`: Number of segments per sequence (default: 10)
- `SEQUENCE_STRIDE`: Sliding window stride in segments (default: 5)

### Segmentation Parameters
- `SEGMENT_DURATION`: Length of each individual segment (seconds)
- `PREICTAL_WINDOW`: Time before seizure for preictal sequences (seconds)
- `SAFE_BUFFER`: Safety margin around seizures (seconds) - **Note: Not currently used**
- `ESTIMATED_FILE_DURATION`: Assumed duration of recording files (seconds)

### Signal Processing
- `LOW_FREQ_HZ`: Bandpass filter lower cutoff (Hz)
- `HIGH_FREQ_HZ`: Bandpass filter upper cutoff (Hz)
- `NOTCH_FREQ_HZ`: Notch filter frequency (Hz)
- `ARTIFACT_THRESHOLD_STD`: Threshold for artifact detection (standard deviations)

### STFT Parameters
- `STFT_NPERSEG`: Window length for STFT
- `STFT_NOVERLAP`: Overlap between STFT windows

### Power Spectrogram
- `LOG_TRANSFORM_EPSILON`: Small value to avoid log(0)
- `APPLY_LOG_TRANSFORM`: Whether to apply log transform

### LSTM Configuration
- `LSTM_HIDDEN_DIM`: LSTM hidden state dimension (default: 256)
- `LSTM_NUM_LAYERS`: Number of stacked LSTM layers (default: 2)
- `LSTM_DROPOUT`: Dropout between LSTM layers (default: 0.3)

### Training Parameters
- `TRAINING_EPOCHS`: Number of epochs
- `BATCH_SIZE`: Batch size (for individual segments - legacy)
- `SEQUENCE_BATCH_SIZE`: Batch size for sequences (default: 16)
- `LEARNING_RATE`: Learning rate for optimizer
- `WEIGHT_DECAY`: L2 regularization
- `NUM_WORKERS`: DataLoader worker processes

### Channel Configuration
- `TARGET_CHANNELS`: List of 18 bipolar EEG channel pairs
- `SKIP_CHANNEL_VALIDATION`: Skip channel validation for faster startup

### Error Handling
- `VERBOSE_WARNINGS`: Print detailed warning messages
- `REQUIRE_MINIMUM_SEGMENTS`: Enforce minimum segments per patient
- `MIN_SEGMENTS_THRESHOLD`: Minimum segment count threshold

### Dataset Path
- `BASE_PATH`: Root directory for CHB-MIT dataset
- `OUTPUT_PREFIX`: File/directory prefix applied to segmentation JSON and preprocessed HDF5 outputs (`all_patients` by default, overrides to the single patient ID when `SINGLE_PATIENT_MODE` is enabled)

## Project Structure

```
├── data_segmentation.py           # Extract SEQUENCES of segments from raw EDF files
├── data_preprocessing.py          # Transform sequences to spectrograms (patient-level split)
├── data_visualization.py          # Visualize sample spectrograms
├── train.py                       # Train CNN-LSTM hybrid model
├── evaluate_test.py              # Evaluate on test set
├── data_segmentation_helpers/
│   ├── config.py                 # Central configuration file
│   ├── parsing.py                # Parse patient summary files
│   ├── segmentation.py           # Segment extraction logic
│   └── channel_mapping.py        # EEG channel utilities
├── preprocessing/
│   ├── data/                     # HDF5 datasets (train/val/test)
│   ├── checkpoints/              # Progress checkpoints
│   ├── logs/                     # Preprocessing logs
│   └── visualizations/           # Sample spectrograms
├── model/
│   ├── epoch_030.pth             # Trained model weights
│   ├── training_metrics.json    # Per-epoch metrics
│   ├── test_results.json        # Test evaluation results
│   └── training_curves.png      # Training/validation curves
├── physionet.org/files/chbmit/1.0.0/  # Raw CHB-MIT dataset
├── all_patients_sequences.json   # Sequence metadata (NEW: sequences instead of segments)
└── chb_mit_analysis.csv          # Dataset statistics
```

## Key Differences from Standard CNN Approach

| Aspect | CNN-Only (Previous) | CNN-LSTM (Current) |
|--------|---------------------|-------------------|
| **Input** | Single 30-second segment | Sequence of 10 segments (5 minutes) |
| **Data Structure** | Individual segments | Temporal sequences |
| **Temporal Modeling** | None (static image) | LSTM across sequence |
| **Train/Test Split** | Random segment split | Patient-level split |
| **Batch Size** | 32 | 16 (sequences are larger) |
| **Output JSON** | `all_patients_segments.json` | `all_patients_sequences.json` |
| **HDF5 Shape** | `(N, 18, freq, time)` | `(N, 10, 18, freq, time)` |
| **Generalization** | To new segments | To **new patients** |
| **Context** | 30 seconds | 5 minutes (10× more) |

## Expected Benefits of CNN-LSTM

- ✅ **Temporal Context**: Models seizure buildup patterns over 5 minutes
- ✅ **Better Generalization**: Patient-level splits ensure model works on unseen patients
- ✅ **Sequential Learning**: LSTM captures temporal dependencies between segments
- ✅ **More Realistic**: Mimics real-world deployment where past EEG informs current prediction
```

## Technologies

- **PyTorch**: Deep learning framework
- **MNE-Python**: EEG data processing and analysis
- **SciPy**: Signal processing (STFT, filtering)
- **HDF5/h5py**: Efficient storage of large arrays
- **scikit-learn**: Data splitting, metrics, preprocessing
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization

## Notes

- Preprocessing is the most time-consuming step (several hours depending on hardware)
- GPU acceleration significantly speeds up training
- The project uses checkpointing throughout to enable resuming from interruptions
- All configuration parameters can be modified in [data_segmentation_helpers/config.py](data_segmentation_helpers/config.py)

## Citation

If you use the CHB-MIT database, please cite:

```
Shoeb, A. (2009). Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment.
PhD Thesis, Massachusetts Institute of Technology.
```

## License

See [LICENSE](LICENSE) file for details.
