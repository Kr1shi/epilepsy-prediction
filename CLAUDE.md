# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

This project implements a seizure prediction pipeline using EEG data from the CHB-MIT Scalp EEG Database. The system uses a **CNN-LSTM hybrid architecture** to classify EEG sequences as preictal (before seizure) or interictal (seizure-free).

### Standard Pipeline Commands

```bash
# 1. Extract sequences from raw EEG files
python data_segmentation.py

# 2. Convert sequences to spectrograms and create train/val/test splits
python data_preprocessing.py

# 3. Train the CNN-LSTM model
python train.py

# 4. Evaluate on test set
python evaluate_test.py

# 5. (Optional) Visualize sample spectrograms before training
python data_visualization.py
```

### Leave-One-Out Cross-Validation (LOOCV) Mode

The pipeline operates in LOOCV mode with a single patient for leave-one-seizure-out cross-validation:

1. Edit `data_segmentation_helpers/config.py`:
   - Set `SINGLE_PATIENT_MODE = True` (enables single-patient mode)
   - Set `SINGLE_PATIENT_ID = "chb06"` (recommended: has 10 documented seizures)
   - Set `LOOCV_FOLD_ID` (0-9 for chb06) to specify which seizure to hold out for testing

2. Run the pipeline normally - it will split data by seizure and use fold-specific random seeds

## Architecture Overview

### Data Flow

```
Raw EEG Files (EDF)
    ↓
[data_segmentation.py] → Extract sequences from single patient with seizure assignments
    ↓ chb06_sequences_prediction.json (LOOCV split: seizure-based)
    ↓
[data_preprocessing.py] → Convert to spectrograms, apply signal processing
    ↓ preprocessing/data/chb06_fold{N}/{train,test}_dataset.h5
    ↓
[train.py] → Train CNN-LSTM on training seizures
    ↓ model/chb06_fold{N}_epoch_XXX.pth
    ↓
[evaluate_test.py] → Test on held-out seizure
    ↓ model/test_results.json
```

### Key Architectural Concepts

**Sequences vs Segments**: The pipeline works with **sequences** of consecutive segments, not individual segments:
- One **segment** = 30 seconds of EEG data from 18 channels
- One **sequence** = 10 consecutive segments (by default, 5 minutes total)
- This allows the LSTM to model **temporal dependencies** across time

**Seizure-Based Splits (LOOCV)**: In LOOCV mode, data is split by seizure index, not randomly. One seizure is held out for testing, while all other seizures are used for training. This ensures the model generalizes to unseen seizure patterns.

**Labeling**: A sequence is labeled based on its **last segment**:
- **Preictal**: Last segment's midpoint is within `PREICTAL_WINDOW` (10 minutes default) before a seizure
- **Interictal**: Last segment is at least `INTERICTAL_BUFFER` (2 hours default) away from any seizure

### CNN-LSTM Model Architecture

```
Input: (batch, sequence_length, 18_channels, freq_bins, time_bins)
    ↓
CNN Feature Extractor (applied to each segment independently):
  - 5 blocks, 16 convolutional layers (VGG-16 style)
  - Extracts 512 features per segment
    ↓
Output per sequence: (batch, sequence_length, 512)
    ↓
LSTM (2-3 layers, hidden_dim=256-512):
  - Models temporal patterns across segments
    ↓
Last hidden state: (batch, hidden_dim)
    ↓
FC Classification Head:
  - hidden_dim → 128 → 2 (preictal/interictal logits)
```

Key characteristics:
- **CNN**: Extracts spatial features from spectrograms (18 channels, frequency-time structure)
- **LSTM**: Captures temporal evolution across the 10-segment sequence
- **Patient-level training**: Prevents data leakage between train/val/test

## Configuration

All parameters are centralized in `data_segmentation_helpers/config.py`. Key sections:

### LOOCV Configuration
- `LOOCV_MODE`: Always `True` (LOOCV is the only supported mode)
- `LOOCV_FOLD_ID`: Fold ID (0-9 for chb06) - seizure index held out for testing
- `LOOCV_TOTAL_SEIZURES`: Total seizures for the patient (10 for chb06)
- `SINGLE_PATIENT_RANDOM_SEED`: Random seed (automatically fold-specific)

### Sequence & Segmentation
- `SEGMENT_DURATION`: Length of each segment (default: 5 seconds)
- `SEQUENCE_LENGTH`: Number of segments per sequence (default: 30)
- `SEQUENCE_STRIDE`: Sliding window stride (default: 5 segments = 83% overlap)
- `PREICTAL_WINDOW`: Time before seizure for preictal label (default: 10 minutes)
- `INTERICTAL_BUFFER`: Minimum distance from seizures for interictal label (default: 2 hours)

### Signal Processing
- `LOW_FREQ_HZ` / `HIGH_FREQ_HZ`: Bandpass filter range (default: 0.5-50 Hz)
- `NOTCH_FREQ_HZ`: Line noise removal (default: 60 Hz)
- `ARTIFACT_THRESHOLD_STD`: MAD-based artifact detection threshold (default: 4 std)
- `STFT_NPERSEG` / `STFT_NOVERLAP`: STFT window settings (default: 256/128)
- `APPLY_LOG_TRANSFORM`: Convert power spectra to dB scale (default: True)

### LSTM Configuration
- `LSTM_HIDDEN_DIM`: Hidden dimension (default: 512)
- `LSTM_NUM_LAYERS`: Stacked layers (default: 3)
- `LSTM_DROPOUT`: Dropout rate (default: 0.5)

### Training
- `TRAINING_EPOCHS`: Number of epochs (default: 15)
- `SEQUENCE_BATCH_SIZE`: Batch size (default: 16)
- `LEARNING_RATE`: Adam LR (default: 0.00001)
- `WEIGHT_DECAY`: L2 regularization (default: 1e-4)
- `PREPROCESSING_WORKERS`: Parallel workers (default: 10, tune for your CPU)

### Channel Configuration
- `TARGET_CHANNELS`: 18 bipolar EEG channels from 10-20 system
- `SKIP_CHANNEL_VALIDATION`: Set to True to skip initial validation (faster startup, but risky)

## Code Structure

```
.
├── data_segmentation.py              # Extract sequences from raw EDF files
├── data_preprocessing.py             # Convert sequences to spectrograms
├── data_visualization.py             # Visualize sample spectrograms
├── train.py                          # Train CNN-LSTM model (uses ResNet18 + LSTM hybrid)
├── evaluate_test.py                  # Evaluate on test set
├── data_segmentation_helpers/
│   ├── config.py                     # Central configuration (modify this to tune experiments)
│   ├── segmentation.py               # Segment/sequence extraction logic
│   └── channel_validation.py         # Validate EDF files have required channels
├── preprocessing/
│   ├── data/                         # Output HDF5 datasets
│   ├── checkpoints/                  # Progress tracking (resume interrupted runs)
│   └── logs/                         # Detailed preprocessing logs
├── model/
│   ├── epoch_*.pth                   # Saved model checkpoints
│   ├── training_metrics.json         # Per-epoch metrics
│   └── training_curves.png           # Visualization of training/validation curves
└── physionet.org/files/chbmit/1.0.0/ # Raw CHB-MIT EEG data (18 patients)
```

## Important Implementation Details

### Sequence Creation (data_segmentation.py)
- Files are processed with a **safety margin** (6s padding) to account for filter edge effects and STFT requirements
- A single patient's seizures are tracked on a **global timeline** accounting for gaps between files
- Boundary sequences (near file edges) are skipped if they don't have complete data
- Sequences are assigned to train/test splits based on LOOCV fold (seizure-based split)

### Signal Processing Pipeline (data_preprocessing.py)
1. **Artifact removal**: Median Absolute Deviation (MAD) based on `ARTIFACT_THRESHOLD_STD`
2. **Bandpass filter**: Remove low and high frequency noise
3. **Notch filter**: Remove 60Hz line noise
4. **Robust normalization**: Per-channel z-score normalization
5. **STFT**: Convert to frequency-time spectrogram
6. **Log transform**: Optional dB conversion for better perceptual scaling

All preprocessing uses checkpointing - interruptions can be resumed without reprocessing completed sequences.

### Model Architecture Notes
- **Custom CNN backbone**: 16-layer VGG-style CNN (not ResNet18) for spatial feature extraction from spectrograms
- **LSTM temporal modeling**: 2-3 stacked LSTM layers to capture temporal patterns across sequence
- **Device handling**: Automatically detects CUDA, MPS (Apple Silicon), or CPU
- **No validation phase**: LOOCV mode trains only on training seizures and evaluates on held-out test seizure
- **Metrics**: Logs training loss, accuracy, precision, recall, F1 per epoch

### Channel Validation
- Validation happens in `data_segmentation.py` (not preprocessing) to fail fast
- Checks that all 18 target channels exist in each EDF file
- Warns about missing channels but continues if `VERBOSE_WARNINGS = True`

## Debugging Tips

- **Check LOOCV fold setup**: Verify `LOOCV_FOLD_ID` is within range (0 to `LOOCV_TOTAL_SEIZURES-1`)
- **Inspect sequence count**: Check `chb06_sequences_prediction.json` for train/test split distribution
- **Monitor training**: Check `model/training_metrics.json` for training convergence (no validation metrics in LOOCV mode)
- **Memory issues**: Reduce `SEQUENCE_BATCH_SIZE` or `PREPROCESSING_WORKERS` if OOM occurs
- **Slow preprocessing**: Increase `PREPROCESSING_WORKERS` (if CPU allows) or `MNE_N_JOBS` for parallel filtering
- **Run all folds**: Loop `LOOCV_FOLD_ID` from 0 to 9 to run complete cross-validation on all seizures
