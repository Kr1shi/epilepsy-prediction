"""Configuration for seizure prediction pipeline

Conv-BiGRU Architecture:
- Single-stream full-band STFT spectrograms
- Conv tower for local spectral patterns → BiGRU for temporal sequence modeling
- Per-patient training with leave-one-seizure-out cross-validation
"""

# =============================================================================
# Task Configuration
# =============================================================================

PREICTAL_WINDOW = 20 * 60  # 20 minutes before seizure (zone starts at -20min)
PREICTAL_ONSET_BUFFER = 0 * 60  # 0 min onset buffer — exclude data too close to seizure
# Effective preictal zone: [-20min, -5min]
INTERICTAL_BUFFER = 30 * 60  # 30 min buffer around seizures (interictal only)

# =============================================================================
# Dataset Configuration
# =============================================================================

BASE_PATH = "physionet.org/files/chbmit/1.0.0/"
ESTIMATED_FILE_DURATION = 3600  # 1 hour (fallback if file duration unavailable)

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Patients to include in processing
PATIENTS = [
    "chb01",  # 7 seizures
    "chb02",  # 3 seizures
    "chb03",  # 7 seizures
    "chb04",  # 4 seizures
    "chb05",  # 5 seizures
    "chb06",  # 10 seizures  ← top 5
    "chb07",  # 3 seizures
    "chb08",  # 5 seizures
    "chb09",  # 4 seizures
    "chb10",  # 7 seizures
    "chb11",  # 3 seizures
    "chb13",  # 12 seizures  ← top 5
    "chb14",  # 8 seizures   ← top 5 (tied)
    "chb15",  # 20 seizures  ← top 5
    "chb16",  # 10 seizures  ← top 5
    "chb17",  # 3 seizures
    "chb18",  # 6 seizures
    "chb19",  # 3 seizures
    "chb20",  # 8 seizures   ← top 5 (tied)
    "chb21",  # 4 seizures
    "chb22",  # 3 seizures
    "chb23",  # 7 seizures
]

ALL_PATIENTS = [
    "chb01",
    "chb02",
    "chb03",
    "chb04",
    "chb05",
    "chb06",
    "chb07",
    "chb08",
    "chb09",
    "chb10",
    "chb11",
    "chb13",
    "chb14",
    "chb15",
    "chb16",
    "chb17",
    "chb18",
    "chb19",
    "chb20",
    "chb21",
    "chb22",
    "chb23",
    "chb24",
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

SEGMENT_DURATION = 5  # seconds per segment
SEQUENCE_LENGTH = 30  # segments per sequence (30 × 5s = 150s = 2.5 min)
SEQUENCE_STRIDE = 6  # segments between sequences (6 × 5s = 30s stride for preictal)

# =============================================================================
# Signal Processing
# =============================================================================

LOW_FREQ_HZ = 0.5
HIGH_FREQ_HZ = 50  # Upper cutoff for clinical EEG bands

# Frequency Bands
FULL_FREQ_BAND = (0.5, 50.0)  # 0.5–50 Hz clinical EEG range
PHASE_FREQ_BAND = (0.0, 5.0)  # Delta/Theta (for Phase)
AMP_FREQ_BAND = (30.0, 50.0)  # Low Gamma (for Amplitude)

STFT_NPERSEG = 104  # STFT window length (~0.4s at 256 Hz, ~2.5 Hz spectral resolution)
STFT_NOVERLAP = 80  # 76.9% overlap (hop = 24 samples)
STFT_NFFT = 256  # FFT size (1.0 Hz bin spacing at 256 Hz; 50 bins after freq mask)

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
# Model Configuration (Conv-GRU)
# =============================================================================

CONV_EMBEDDING_DIM = 128  # Conv tower output dimension
GRU_HIDDEN_DIM = 64  # BiGRU hidden size (output = 2x this for bidirectional)
GRU_NUM_LAYERS = 2  # Number of stacked GRU layers
GRU_DROPOUT = 0.3  # Dropout for GRU + FC head

# =============================================================================
# Training Configuration
# =============================================================================

PRETRAINING_EPOCHS = 10
TRAINING_EPOCHS = 15
SEQUENCE_BATCH_SIZE = 4
LEARNING_RATE = 1e-4  # Pretraining LR
FINETUNING_LEARNING_RATE = 3e-4  # Fine-tuning LR (transformer layer + FC head)
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
        raise ValueError(
            f"patient_index must be 0-{n_patients - 1}, got {patient_index}"
        )

    patient_id = PATIENTS[patient_index]

    return {
        "patient_id": patient_id,
        "random_seed": 42 + patient_index,
        "output_prefix": f"{patient_id}",
    }
