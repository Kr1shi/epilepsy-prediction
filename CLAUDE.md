# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

This project implements a seizure prediction pipeline using EEG data from the CHB-MIT Scalp EEG Database. The system uses a **Conv-Transformer architecture** to classify 30-minute EEG windows as preictal (before seizure) or interictal (seizure-free). Training uses **cross-patient pretraining** followed by **per-patient fine-tuning**.

### Standard Pipeline Commands

```bash
# 1. Extract sequences from raw EEG files
python data_segmentation.py

# 2. Convert sequences to single-stream STFT spectrograms and save as HDF5
python data_preprocessing.py

# 3. Cross-patient pretraining (train shared encoder on ALL patients)
python train.py --pretrain

# 4. Per-patient fine-tuning (loads pretrained weights automatically)
python train.py

# 5. Evaluate on test set
python evaluate_test.py
```

### Patient Selection

Edit `data_segmentation_helpers/config.py`:
- `PATIENTS`: list of patient IDs to process (e.g. `["chb06", "chb13", ...]`)
- `PATIENT_INDEX`: set to an integer to process a single patient, or `None` to process all

All pipeline scripts respect these settings and loop over the selected patients.

## Architecture Overview

### Data Flow

```
Raw EEG Files (EDF)
    ↓
[data_segmentation.py] → Extract 30-min sequences with randomized train/val/test splits
    ↓ {patient_id}_sequences_{task_mode}.json
    ↓
[data_preprocessing.py] → Single-stream full-band STFT spectrograms, global z-score normalization
    ↓ preprocessing/data/{patient_id}/{train,val,test}_dataset.h5
    ↓
[train.py --pretrain] → Cross-patient pretraining on ALL patients
    ↓ model/pretrained_encoder.pth
    ↓
[train.py] → Per-patient fine-tuning (loads pretrained weights)
    ↓ model/{patient_id}/best_model.pth
    ↓
[evaluate_test.py] → Evaluate on test set, generate per-seizure plots
    ↓ model/{patient_id}/test_results.json
    ↓ result_plots/{patient_id}/seizure_XX.png
```

### Key Architectural Concepts

**Sequences vs Segments**:
- One **segment** = 5 seconds of EEG data from 18 channels
- One **sequence** = 360 consecutive segments (30 minutes total)
- The Transformer models **long-range temporal dependencies** across all 360 segments

**Preictal Zone with Onset Buffer**:
- **Preictal zone**: [-40 min, -10 min] before seizure onset
- `PREICTAL_WINDOW = 40 * 60` defines the zone start (40 min before seizure)
- `PREICTAL_ONSET_BUFFER = 10 * 60` defines the zone end (10 min before seizure)
- The 10-min buffer ensures the model predicts seizures with enough clinical lead time

**Leave-One-Seizure-Out Split**:
- One seizure is randomly held out for **test** — all its positive sequences go to the test set
- Remaining seizures' positive sequences are split into **train/val** proportionally
- Interictal sequences are distributed across splits by ratio
- Each split is independently class-balanced
- Falls back to randomized split if a patient has fewer than 2 seizures
- Split ratios configured via `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO` in config.py

**Adaptive Sliding Window**:
- Positive regions (preictal): **overlapping stride** (`SEQUENCE_STRIDE=12` segments = 1-min stride, ~97% overlap) to maximize positive samples (~30 per seizure)
- Interictal regions: **non-overlapping stride** (full sequence length) to prevent near-duplicate interictal samples

**Class Balancing**: Interictal sequences are downsampled per split to `INTERICTAL_TO_PREICTAL_RATIO × n_positive` (default 1:1).

**Cross-Patient Pretraining**: A shared encoder is trained on combined data from ALL patients before per-patient fine-tuning. This compensates for limited per-patient data (~30 preictal samples per seizure).

### Conv-Transformer Model Architecture

```
Input: (batch=4, sequence_length=360, 18_channels, 128_freq_bins, 9_time_bins)
    ↓
For each of the 360 segments independently:
  ConvTower:
    Block1: Conv2d(18→32, k=3, pad=1) + BN + GELU + MaxPool2d(2)  → (B, 32, 64, 4)
    Block2: Conv2d(32→64, k=3, pad=1) + BN + GELU + MaxPool2d(2)  → (B, 64, 32, 2)
    Block3: Conv2d(64→128, k=3, pad=1) + BN + GELU               → (B, 128, 32, 2)
    AdaptiveAvgPool2d(1,1) → Flatten                               → (B, 128)
    ↓
Reshape to sequence: (batch, 360, 128)
    ↓
Prepend [CLS] token: (batch, 361, 128)
Add learned positional embeddings
    ↓
TransformerEncoder (2 layers, 4 heads, pre-norm, GELU, ffn_dim=512):
    ↓
LayerNorm → extract [CLS] output: (batch, 128)
    ↓
FC head: Linear(128→64) + GELU + Dropout(0.3) → Linear(64→2)
    ↓
Output: (batch, 2)  [interictal / preictal logits]
```

Total parameters: ~547K (ConvTower ~98K, Transformer ~394K, CLS+pos ~46K, FC ~9K).

## Configuration

All parameters are in `data_segmentation_helpers/config.py`.

### Task
- `TASK_MODE`: `"prediction"` (preictal vs interictal) or `"detection"` (ictal vs interictal)
- `PREICTAL_WINDOW`: zone start before seizure (default: `40 * 60` = 40 min)
- `PREICTAL_ONSET_BUFFER`: zone end before seizure (default: `10 * 60` = 10 min)
- `INTERICTAL_BUFFER`: exclusion zone around each seizure for interictal (default: `1 * 60 * 60` = 1 hour)
- `INTERICTAL_TO_PREICTAL_RATIO`: interictal downsampling ratio (default: `1.0`)

### Dataset
- `BASE_PATH`: path to raw CHB-MIT EDF files
- `PATIENTS`: list of patient IDs to process
- `PATIENT_INDEX`: single patient index or `None` for all
- `TRAIN_RATIO` / `VAL_RATIO` / `TEST_RATIO`: split ratios (default: 0.8 / 0.1 / 0.1)

### Sequence & Segmentation
- `SEGMENT_DURATION`: seconds per segment (default: `5`)
- `SEQUENCE_LENGTH`: segments per sequence (default: `360` → 30 min total)
- `SEQUENCE_STRIDE`: stride in segments for positive-region sliding window (default: `12` → 1 min stride)

### Signal Processing (Single-Stream)
- `FULL_FREQ_BAND`: frequency range for full-band STFT (default: `(0.5, 128.0)` Hz)
- `NOTCH_FREQ_HZ`: line noise removal (default: `60` Hz)
- `STFT_NPERSEG` / `STFT_NOVERLAP` / `STFT_NFFT`: STFT settings (default: `256` / `128` / `256`, 1.0 Hz resolution)
- `ARTIFACT_THRESHOLD_STD`: MAD-based artifact threshold (default: `4`)
- `APPLY_LOG_TRANSFORM`: log10 of power spectra (default: `True`)

### Model (Conv-Transformer)
- `CONV_EMBEDDING_DIM`: Conv tower output / Transformer d_model (default: `128`)
- `TRANSFORMER_NUM_LAYERS`: Transformer encoder layers (default: `2`)
- `TRANSFORMER_NUM_HEADS`: attention heads (default: `4`, head_dim = 32)
- `TRANSFORMER_FFN_DIM`: feedforward hidden dimension (default: `512`)
- `TRANSFORMER_DROPOUT`: dropout rate (default: `0.3`)
- `USE_CLS_TOKEN`: CLS token pooling vs mean pooling (default: `True`)

### Training
- `TRAINING_EPOCHS`: number of epochs (default: `30`)
- `SEQUENCE_BATCH_SIZE`: batch size (default: `4`, reduced for 30-min windows)
- `LEARNING_RATE`: Adam LR (default: `1e-4`)
- `WEIGHT_DECAY`: L2 regularization (default: `1e-4`)
- `NUM_WORKERS`: DataLoader workers (default: `0`, must be 0 for lazy HDF5 loading)
- `MNE_N_JOBS`: parallel jobs for MNE filtering (default: `8`)

### Channel Configuration
- `TARGET_CHANNELS`: 18 bipolar EEG channels from the 10-20 system
- `SKIP_CHANNEL_VALIDATION`: skip per-file channel checks (default: `False`)

## Code Structure

```
.
├── data_segmentation.py              # Stage 1: Extract sequences from raw EDF files
├── data_preprocessing.py             # Stage 2: Single-stream STFT spectrograms → HDF5
├── train.py                          # Stage 3: Conv-Transformer training (pretrain + fine-tune)
├── evaluate_test.py                  # Stage 4: Evaluate, smooth predictions, plot dynamics
├── total_accuracy.py                 # Per-seizure accuracy & false positive rate calculation
├── extract_and_visualize_metrics.py  # Offline batch metrics visualization
├── data_segmentation_helpers/
│   ├── config.py                     # Central configuration — edit to tune experiments
│   ├── segmentation.py               # Summary file parsing and sequence extraction logic
│   ├── channel_validation.py         # Validate EDF files have all required channels
│   └── seizure_counts.py             # Precomputed seizure counts per patient
├── preprocessing/
│   ├── data/{patient_id}/            # Final HDF5 datasets (train/val/test_dataset.h5)
│   ├── intermediate/{patient_id}/    # Intermediate HDF5 before global normalization
│   ├── checkpoints/{patient_id}/     # Progress tracking (resume interrupted runs)
│   └── logs/{patient_id}/            # Detailed preprocessing logs
├── model/
│   ├── pretrained_encoder.pth        # Cross-patient pretrained weights
│   └── {patient_id}/
│       ├── best_model.pth            # Best fine-tuned checkpoint (by val AUC)
│       ├── training_metrics.json     # Per-epoch train/val metrics
│       └── training_curves.png       # Training/validation curves plot
├── result_plots/{patient_id}/
│   └── seizure_XX.png                # Per-seizure probability dynamics plots
└── physionet.org/files/chbmit/1.0.0/ # Raw CHB-MIT EEG data
```

## Important Implementation Details

### Sequence Creation (data_segmentation.py)
- Files are processed with a **6-second safety margin** at both ends to account for filter edge effects and STFT padding requirements
- A patient's seizures are tracked on a **global timeline** that accounts for real-time gaps between recording files
- **Two-pass labeling**: (1) check for positive class (preictal/ictal), (2) check buffer zones only if not already positive — buffer zones never override a positive label
- **Onset buffer exclusion**: the zone between `seizure_start - PREICTAL_ONSET_BUFFER` and `seizure_end + INTERICTAL_BUFFER` is excluded from interictal sequences
- All sequences are shuffled and randomly split into train/val/test at the sequence level (80/10/10 default), then each split is independently class-balanced

### Signal Processing Pipeline (data_preprocessing.py)
1. **Notch filter**: remove 60 Hz line noise (MNE)
2. **Artifact removal**: MAD-based per-channel detection + linear interpolation of artifacts
3. **STFT**: `nfft=256` for 1.0 Hz frequency resolution at 256 Hz sampling rate
4. **Full-band power**: `|z|²` of complex STFT coefficients in 0.5–128 Hz band (128 frequency bins)
5. **Log transform**: `log10(power + epsilon)` applied to all power values
6. **Global z-score normalization**: mean/std computed from training set, applied to all splits

HDF5 format: `spectrograms` dataset with shape `(N, 360, 18, 128, 9)`. All preprocessing is checkpointed.

### Training (train.py)
- **Lazy HDF5 loading**: each sample is ~30MB, loaded on-demand via `h5py` in `__getitem__()`. Requires `num_workers=0`.
- **Cross-patient pretraining** (`--pretrain`): combines train datasets from ALL patients via `ConcatDataset`, trains shared encoder, saves `model/pretrained_encoder.pth`
- **Per-patient fine-tuning**: loads pretrained weights if available, fine-tunes on single patient data
- Best model checkpoint selected by val AUC, stores `optimal_threshold` and smoothing params
- Optimizer: Adam with `StepLR(step_size=5, gamma=0.5)`
- Loss: FocalLoss

### Evaluation (evaluate_test.py)
- Applies a sliding-window majority vote smoother for prediction smoothing
- Per-seizure accuracy: fraction of ground-truth seizure events where ≥1 smoothed prediction fires
- False positive rate: smoothed interictal false positives normalized to events/hour
- Probability dynamics plots sorted chronologically using HDF5 metadata

### Channel Validation
- Validation happens in `data_segmentation.py` (fail fast before any preprocessing)
- Files missing required channels are excluded; their seizures are also excluded from the global timeline

## Tensor Shape Summary

| Stage | Shape |
|-------|-------|
| Raw EEG segment | (18, 1280) — 5s × 256Hz |
| STFT → freq mask | (18, 128, 9) — 128 freq bins at 1Hz |
| log10(power) | (18, 128, 9) |
| Per window (HDF5) | (360, 18, 128, 9) — 30 min |
| Batch | (4, 360, 18, 128, 9) |
| Conv tower (flat) | (1440, 128) |
| Sequence + CLS | (4, 361, 128) |
| Transformer out | (4, 361, 128) |
| CLS → FC | (4, 2) |

## Known Issues / Limitations

- **Leave-one-seizure-out test split** ensures the test seizure is never seen during training, providing rigorous evaluation
- **Near-duplicate positive samples**: ~97% sequence overlap in preictal regions creates many highly similar training examples
- **Large sample size**: each 30-min window is ~30MB, requiring lazy HDF5 loading and small batch sizes
- **`num_workers` must be 0**: HDF5 lazy loading is not fork-safe

## Debugging Tips

- **Inspect sequence counts**: Check `{patient_id}_sequences_prediction.json` → `summary` field for train/val/test distribution
- **Preprocessing resume**: Delete `preprocessing/checkpoints/{patient_id}/progress.json` to force a full reprocess
- **Memory issues**: Reduce `SEQUENCE_BATCH_SIZE` (default 4)
- **Slow preprocessing**: Increase `MNE_N_JOBS` (parallel MNE filtering)
- **Single patient run**: Set `PATIENT_INDEX = 0` (or any index) in config.py
- **Old format error**: If you see "Old format" errors, delete `preprocessing/` and re-run `data_preprocessing.py`
