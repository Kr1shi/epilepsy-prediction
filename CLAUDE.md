# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

This project implements a seizure prediction pipeline using EEG data from the CHB-MIT Scalp EEG Database. The system uses a **Dual-Stream CNN-LSTM hybrid architecture** to classify EEG sequences as preictal (before seizure) or interictal (seizure-free), trained and evaluated per patient.

### Standard Pipeline Commands

```bash
# 1. Extract sequences from raw EEG files
python data_segmentation.py

# 2. Convert sequences to dual-stream spectrograms and save as HDF5
python data_preprocessing.py

# 3. Train the Dual-Stream CNN-LSTM model
python train.py

# 4. Evaluate on test set
python evaluate_test.py
```

### Patient Selection

Edit `data_segmentation_helpers/config.py`:
- `PATIENTS`: list of patient IDs to process (e.g. `["chb01", "chb02", ...]`)
- `PATIENT_INDEX`: set to an integer to process a single patient, or `None` to process all

All pipeline scripts respect these settings and loop over the selected patients.

## Architecture Overview

### Data Flow

```
Raw EEG Files (EDF)
    ↓
[data_segmentation.py] → Extract sequences per patient with seizure-based train/test splits
    ↓ {patient_id}_sequences_{task_mode}.json
    ↓
[data_preprocessing.py] → Dual-stream STFT spectrograms, global normalization
    ↓ preprocessing/data/{patient_id}/{train,test}_dataset.h5
    ↓
[train.py] → Train Dual-Stream CNN-LSTM model
    ↓ model/{patient_id}/best_model.pth
    ↓
[evaluate_test.py] → Evaluate on test seizure(s), generate plots
    ↓ model/{patient_id}/test_results.json
    ↓ result_plots/{patient_id}/seizure_XX.png
```

### Key Architectural Concepts

**Sequences vs Segments**: The pipeline works with **sequences** of consecutive segments:
- One **segment** = 5 seconds of EEG data from 18 channels
- One **sequence** = 30 consecutive segments (150 seconds total by default)
- The LSTM models **temporal dependencies** across segments within a sequence

**Randomized Sequence-Level Split**:
- All sequences are shuffled and randomly assigned to **train/val/test** using configurable ratios (default 80/10/10)
- Sequences from the same seizure may appear in different splits
- Split ratios configured via `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO` in config.py

**Labeling**: A sequence is labeled based on its **last segment**:
- **Preictal**: last segment falls within `PREICTAL_WINDOW` (10 min default) before a seizure
- **Interictal**: sequence does not overlap with any buffer zone around seizures (`INTERICTAL_BUFFER` = 30 min default)
- Buffer zones around each seizure are excluded from interictal to avoid ambiguous labels

**Adaptive Sliding Window**:
- Positive regions (preictal/ictal): **overlapping stride** (`SEQUENCE_STRIDE` segments = 83% overlap by default) to maximise positive samples
- Interictal regions: **non-overlapping stride** (full sequence length) to prevent near-duplicate interictal samples

**Class Balancing**: Interictal sequences are downsampled per split to `INTERICTAL_TO_PREICTAL_RATIO × n_positive` (default 2:1). Balancing is done independently within train and test.

### Dual-Stream CNN-LSTM Model Architecture

```
Input: (batch, sequence_length=30, 18_channels, freq_bins, time_bins)
    ↓
For each segment independently:
  DualStreamSpectrogramEncoder:
    ├── Phase Tower (CompactEEGCNN): processes 0.5–12 Hz phase spectrogram
    │     Block1: Conv2d(18→32) + BN + ReLU + Dropout2d + MaxPool2d
    │     Block2: Conv2d(32→64) + BN + ReLU + Dropout2d + MaxPool2d
    │     Block3: Conv2d(64→64) + BN + ReLU + Dropout2d
    │     AdaptiveAvgPool → (batch, 64, 1, time)  [frequency pooled to 1]
    │
    └── Amplitude Tower (CompactEEGCNN): processes 20–128 Hz amplitude spectrogram
          (same structure as Phase Tower)
    ↓
  Concatenate along channel dim → (batch, 128, 1, time)
  1×1 Fusion Conv: Conv2d(128→256) + BN + ReLU + Dropout
  Global Time Pooling → (batch, 256)          [one embedding per segment]
    ↓
Reshape to sequence: (batch, sequence_length=5, 256)
    ↓
BiLSTM (hidden=512, layers=3, bidirectional):
    ↓
Attention pooling over sequence → (batch, 1024)
    ↓
FC head: Linear(1024→64) + ReLU + Dropout → Linear(64→2)
    ↓
Output: (batch, 2)  [interictal / preictal logits]
```

Total parameters: ~1.2M (BiLSTM ~0.8M, CNNs ~0.3M, attention+FC ~0.1M).

## Configuration

All parameters are in `data_segmentation_helpers/config.py`.

### Task
- `TASK_MODE`: `"prediction"` (preictal vs interictal) or `"detection"` (ictal vs interictal)
- `PREICTAL_WINDOW`: seconds before seizure labelled preictal (default: `10 * 60` = 10 min)
- `INTERICTAL_BUFFER`: exclusion zone around each seizure for interictal (default: `30 * 60` = 30 min)
- `INTERICTAL_TO_PREICTAL_RATIO`: interictal downsampling ratio (default: `2.0`)

### Dataset
- `BASE_PATH`: path to raw CHB-MIT EDF files
- `PATIENTS`: list of patient IDs to process
- `PATIENT_INDEX`: single patient index or `None` for all

### Sequence & Segmentation
- `SEGMENT_DURATION`: seconds per segment (default: `5`)
- `SEQUENCE_LENGTH`: segments per sequence (default: `30` → 150 s total)
- `SEQUENCE_STRIDE`: stride in segments for positive-region sliding window (default: `5` → 83% overlap)

### Signal Processing (Dual-Stream)
- `PHASE_FREQ_BAND`: frequency range for phase stream (default: `(0.5, 12.0)` Hz — delta/theta)
- `AMP_FREQ_BAND`: frequency range for amplitude stream (default: `(20.0, 128.0)` Hz — gamma/HFO)
- `NOTCH_FREQ_HZ`: line noise removal (default: `60` Hz)
- `STFT_NPERSEG` / `STFT_NOVERLAP`: STFT window settings (default: `256` / `128`, 50% overlap)
- `ARTIFACT_THRESHOLD_STD`: MAD-based artifact threshold (default: `4`)
- `APPLY_LOG_TRANSFORM`: log10 of amplitude power spectra (default: `True`)

### Model
- `LSTM_HIDDEN_DIM`: BiLSTM hidden size per direction (default: `128`)
- `LSTM_NUM_LAYERS`: stacked LSTM layers (default: `2`)
- `LSTM_DROPOUT`: dropout between LSTM layers (default: `0.5`)

### Training
- `TRAINING_EPOCHS`: number of epochs (default: `5`)
- `SEQUENCE_BATCH_SIZE`: batch size (default: `16`)
- `LEARNING_RATE`: Adam LR (default: `1e-4`)
- `WEIGHT_DECAY`: L2 regularization (default: `1e-4`)
- `NUM_WORKERS`: DataLoader workers (default: `4`)
- `MNE_N_JOBS`: parallel jobs for MNE filtering (default: `8`)

### Channel Configuration
- `TARGET_CHANNELS`: 18 bipolar EEG channels from the 10-20 system
- `SKIP_CHANNEL_VALIDATION`: skip per-file channel checks (default: `False`)

## Code Structure

```
.
├── data_segmentation.py              # Stage 1: Extract sequences from raw EDF files
├── data_preprocessing.py             # Stage 2: Dual-stream STFT spectrograms → HDF5
├── train.py                          # Stage 3: Train Dual-Stream CNN-LSTM
├── evaluate_test.py                  # Stage 4: Evaluate, smooth predictions, plot dynamics
├── total_accuracy.py                 # Per-seizure accuracy & false positive rate calculation
├── extract_and_visualize_metrics.py  # Offline batch metrics visualization (hardcoded paths)
├── data_segmentation_helpers/
│   ├── config.py                     # Central configuration — edit to tune experiments
│   ├── segmentation.py               # Summary file parsing and sequence extraction logic
│   ├── channel_validation.py         # Validate EDF files have all required channels
│   └── seizure_counts.py             # Precomputed seizure counts per patient
├── preprocessing/
│   ├── data/{patient_id}/            # Final HDF5 datasets (train_dataset.h5, test_dataset.h5)
│   ├── intermediate/{patient_id}/    # Intermediate HDF5 before global normalization
│   ├── checkpoints/{patient_id}/     # Progress tracking (resume interrupted runs)
│   └── logs/{patient_id}/            # Detailed preprocessing logs
├── model/{patient_id}/
│   ├── best_model.pth                # Best checkpoint (by val AUC)
│   ├── training_metrics.json         # Per-epoch train/val metrics
│   └── training_curves.png           # Training/validation curves plot
├── result_plots/{patient_id}/
│   └── seizure_XX.png                # Per-seizure probability dynamics plots
└── physionet.org/files/chbmit/1.0.0/ # Raw CHB-MIT EEG data
```

## Important Implementation Details

### Sequence Creation (data_segmentation.py)
- Files are processed with a **6-second safety margin** at both ends to account for filter edge effects and STFT padding requirements
- A patient's seizures are tracked on a **global timeline** that accounts for real-time gaps between recording files
- **Two-pass labeling**: (1) check for positive class (preictal/ictal), (2) check buffer zones only if not already positive — buffer zones never override a positive label
- All sequences are shuffled and randomly split into train/val/test at the sequence level (80/10/10 default), then each split is independently class-balanced

### Signal Processing Pipeline (data_preprocessing.py)
1. **Notch filter**: remove 60 Hz line noise (MNE)
2. **Artifact removal**: MAD-based per-channel detection + linear interpolation of artifacts
3. **STFT**: `nfft = sfreq / 0.1` for 0.1 Hz frequency resolution
4. **Phase stream**: `np.angle()` of complex STFT coefficients in 0.5–12 Hz band — no normalization applied
5. **Amplitude stream**: `|z|²` of STFT in 20–128 Hz band, stride-decimated ×10 in frequency, log10 transformed, then globally z-score normalized using train-set mean/std

All preprocessing is checkpointed — interrupted runs resume without reprocessing completed files.

### Training (train.py)
- **Val split**: `val_loader` uses randomly assigned val sequences. `best_model.pth` is selected by val AUC
- Best model checkpoint also stores `optimal_threshold` (Youden's J on train set) and `smoothing_window` / `smoothing_count` for post-hoc smoothing
- Optimizer: Adam with `StepLR(step_size=5, gamma=0.5)`
- Loss: `CrossEntropyLoss` without class weights

### Evaluation (evaluate_test.py)
- Applies a sliding-window majority vote smoother: predicts positive if ≥ `X` positives appear in a window of `T` consecutive predictions
- Smoothing params are loaded from checkpoint, or can be overridden with `MANUAL_SMOOTHING_WINDOW` / `MANUAL_SMOOTHING_COUNT`
- Per-seizure accuracy: fraction of ground-truth seizure events where ≥1 smoothed prediction fires within the event window
- False positive rate: smoothed interictal false positives normalized to events/hour
- Probability dynamics plots are sorted chronologically using HDF5 metadata before plotting

### Channel Validation
- Validation happens in `data_segmentation.py` (fail fast before any preprocessing)
- Files missing required channels are excluded; their seizures are also excluded from the global timeline

## Known Issues / Limitations

- **Sequence-level random split** means sequences from the same seizure may appear in multiple splits — this is intentional for maximizing data utilization
- **Near-duplicate positive samples**: 83% sequence overlap in preictal regions creates many highly similar training examples, which accelerates overfitting
- **`INTERICTAL_BUFFER` is 30 minutes** (not 2 hours as older docs stated) — some interictal sequences may contain subtle preictal activity
- **Amplitude frequency decimation** (`::10`) is done without an anti-aliasing filter
- **No data augmentation** is applied during training

## Debugging Tips

- **Inspect sequence counts**: Check `{patient_id}_sequences_prediction.json` → `summary` field for train/test split distribution
- **Preprocessing resume**: Delete `preprocessing/checkpoints/{patient_id}/progress.json` to force a full reprocess
- **Memory issues**: Reduce `SEQUENCE_BATCH_SIZE` or `NUM_WORKERS`
- **Slow preprocessing**: Increase `MNE_N_JOBS` (parallel MNE filtering) or reduce `SEQUENCE_LENGTH`
- **Single patient run**: Set `PATIENT_INDEX = 0` (or any index) in config.py to run just one patient