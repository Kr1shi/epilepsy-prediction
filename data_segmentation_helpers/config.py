"""Configuration for seizure prediction pipeline

Conv-Transformer Architecture:
- Single-stream full-band STFT spectrograms
- Conv tower for local spectral patterns → Transformer for long-range temporal attention
- Cross-patient pretraining, then per-patient fine-tuning
"""

# =============================================================================
# Task Configuration
# =============================================================================

TASK_MODE = "prediction"  # 'prediction' (preictal vs interictal) or 'detection' (ictal vs interictal)
PREICTAL_WINDOW = 40 * 60  # 40 minutes before seizure (zone starts at -40min)
PREICTAL_ONSET_BUFFER = 10 * 60  # 10 minutes before seizure (zone ends at -10min)
                                  # Effective preictal zone: [-40min, -10min]
INTERICTAL_BUFFER = 1 * 60 * 60  # 1 hour buffer around seizures

# =============================================================================
# Dataset Configuration
# =============================================================================

BASE_PATH = "physionet.org/files/chbmit/1.0.0/"
ESTIMATED_FILE_DURATION = 3600  # 1 hour (fallback if file duration unavailable)
INTERICTAL_TO_PREICTAL_RATIO = 1.0  # Balanced 1:1 ratio of interictal to preictal sequences

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Patients to include in processing
PATIENTS = [
    "chb01",   # 7 seizures
    "chb02",   # 3 seizures
    "chb03",   # 7 seizures
    "chb04",   # 4 seizures
    "chb05",   # 5 seizures
    "chb06",    # 10 seizures  ← top 5
    "chb07",   # 3 seizures
    "chb08",   # 5 seizures
    "chb09",   # 4 seizures
    "chb10",   # 7 seizures
    "chb11",   # 3 seizures
    "chb13",    # 12 seizures  ← top 5
    "chb14",    # 8 seizures   ← top 5 (tied)
    "chb15",    # 20 seizures  ← top 5
    "chb16",    # 10 seizures  ← top 5
    "chb17",   # 3 seizures
    "chb18",   # 6 seizures
    "chb19",   # 3 seizures
    "chb20",    # 8 seizures   ← top 5 (tied)
    "chb21",   # 4 seizures
    "chb22",   # 3 seizures
    "chb23",   # 7 seizures
]

ALL_PATIENTS = [
    "chb01", "chb02", "chb03", "chb04",
    "chb05", "chb06", "chb07", "chb08",
    "chb09", "chb10", "chb11", "chb13",
    "chb14", "chb15", "chb16", "chb17",
    "chb18", "chb19", "chb20", "chb21",
    "chb22", "chb23", "chb24"
]

# Current patient index to process (0 to len(PATIENTS)-1, or None for all)
# PATIENT_INDEX = None
PATIENT_INDEX = None
TEST_SEIZURE = None

# Current fold to process (0 to len(LOPO_PATIENTS)-1, or None for all folds)
# LOPO_FOLD_ID = 2
LOPO_FOLD_ID = None

# Precomputed seizure counts (for reference)
from data_segmentation_helpers.seizure_counts import SEIZURE_COUNTS

# =============================================================================
# Sequence Configuration
# =============================================================================

SEGMENT_DURATION = 5    # seconds per segment
SEQUENCE_LENGTH = 360   # segments per sequence (360 × 5s = 1800s = 30 min)
SEQUENCE_STRIDE = 12    # segments between sequences (12 × 5s = 60s = 1 min stride for preictal)

# =============================================================================
# Signal Processing
# =============================================================================

LOW_FREQ_HZ = 0.5
HIGH_FREQ_HZ = 128  # Extended to capture High Gamma/HFO
NOTCH_FREQ_HZ = 60

# Frequency Bands
FULL_FREQ_BAND = (0.5, 128.0)  # Full-band for single-stream STFT
PHASE_FREQ_BAND = (0.0, 5.0)  # Delta/Theta (for Phase)
AMP_FREQ_BAND = (80.0, 128.0)  # Gamma/HFO (for Amplitude)

STFT_NPERSEG = 256   # STFT window length
STFT_NOVERLAP = 128  # STFT overlap (50%)
STFT_NFFT = 256      # FFT size (1.0 Hz resolution at 256 Hz sampling rate)

ARTIFACT_THRESHOLD_STD = 4  # MAD threshold for artifact removal
LOG_TRANSFORM_EPSILON = 1e-12  # Avoid log(0)
APPLY_LOG_TRANSFORM = True

TARGET_CHANNELS = [
    "C3-P3",
    "C4-P4",
    "CZ-PZ",
    "F3-C3",
    "F4-C4",
    "F7-T7",
    "F8-T8",
    "FP1-F3",
    "FP1-F7",
    "FP2-F4",
    "FP2-F8",
    "FZ-CZ",
    "P3-O1",
    "P4-O2",
    "P7-O1",
    "P8-O2",
    "T7-P7",
    "T8-P8",
]

# =============================================================================
# Model Configuration (Conv-Transformer)
# =============================================================================

CONV_EMBEDDING_DIM = 128         # Conv tower output / Transformer d_model
TRANSFORMER_NUM_LAYERS = 1       # Transformer encoder layers (reduced from 2)
TRANSFORMER_NUM_HEADS = 2        # Attention heads (head_dim = 128/2 = 64)
TRANSFORMER_FFN_DIM = 256        # Feedforward hidden dimension (reduced from 512)
TRANSFORMER_DROPOUT = 0.3        # Dropout for Transformer + FC head
USE_CLS_TOKEN = True             # CLS token pooling (vs mean pooling)

# =============================================================================
# Training Configuration
# =============================================================================

PRETRAINING_EPOCHS = 10
TRAINING_EPOCHS = 5
SEQUENCE_BATCH_SIZE = 16
LEARNING_RATE = 1e-4             # Pretraining LR
FINETUNING_LEARNING_RATE = 5e-4  # Higher LR for fine-tuning (FC head only)
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

# =============================================================================
# Performance Settings
# =============================================================================

PREPROCESSING_WORKERS = 10  # Parallel workers for preprocessing
MNE_N_JOBS = 8  # Parallel jobs for MNE filtering
VERBOSE_WARNINGS = False  # Suppress non-critical warnings
SKIP_CHANNEL_VALIDATION = False

# =============================================================================
# Helper Functions
# =============================================================================


def get_patient_config(patient_index):
    """Get configuration for a specific patient.

    Args:
        patient_index: Index in PATIENTS list (0 to len(PATIENTS)-1)

    Returns:
        dict with: patient_id, random_seed, output_prefix
    """
    n_patients = len(PATIENTS)
    if patient_index < 0 or patient_index >= n_patients:
        raise ValueError(f"patient_index must be 0-{n_patients-1}, got {patient_index}")

    patient_id = PATIENTS[patient_index]

    return {
        "patient_id": patient_id,
        "random_seed": 42 + patient_index,
        "output_prefix": f"{patient_id}",
    }
