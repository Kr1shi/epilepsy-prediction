"""Configuration for seizure prediction pipeline

Leave-One-Patient-Out (LOPO) cross-validation:
- One patient is held out for testing
- All other patients are used for training
"""

# =============================================================================
# Task Configuration
# =============================================================================

TASK_MODE = "prediction"  # 'prediction' (preictal vs interictal) or 'detection' (ictal vs interictal)
PREICTAL_WINDOW = 10 * 60  # 10 minutes before seizure
INTERICTAL_BUFFER = 120 * 60  # 2 hours buffer from seizures

# =============================================================================
# Dataset Configuration
# =============================================================================

BASE_PATH = "physionet.org/files/chbmit/1.0.0/"
ESTIMATED_FILE_DURATION = 3600  # 1 hour (fallback if file duration unavailable)

# Patients to include in cross-validation (one fold per patient)
LOPO_PATIENTS = [
    "chb01",
    "chb02",
    # "chb05",
    # "chb06",
    # "chb07",
    # "chb08",
    # "chb11",
    # "chb13",
    # "chb14",
    # "chb15",
    # "chb17",
    # "chb19",
    # "chb20",
    # "chb23",
    # "chb24",
]

# Current fold to process (0 to len(LOPO_PATIENTS)-1, or None for all folds)
LOPO_FOLD_ID = None

# Whether to include other patients in the training set (True = standard LOPO, False = single patient split)
LOPO_INCLUDE_OTHER_PATIENTS = True

# Precomputed seizure counts (for reference)
from data_segmentation_helpers.seizure_counts import SEIZURE_COUNTS

# =============================================================================
# Sequence Configuration
# =============================================================================

SEGMENT_DURATION = 5  # seconds per segment
SEQUENCE_LENGTH = 30  # segments per sequence (5 min total)
SEQUENCE_STRIDE = 5  # segments between sequences (83% overlap in preictal)

# =============================================================================
# Signal Processing
# =============================================================================

LOW_FREQ_HZ = 0.5
HIGH_FREQ_HZ = 50
NOTCH_FREQ_HZ = 60

STFT_NPERSEG = 256  # STFT window length
STFT_NOVERLAP = 128  # STFT overlap (50%)

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
# Model Configuration
# =============================================================================

LSTM_HIDDEN_DIM = 512
LSTM_NUM_LAYERS = 3
LSTM_DROPOUT = 0.5

# =============================================================================
# Training Configuration
# =============================================================================

TRAINING_EPOCHS = 5
SEQUENCE_BATCH_SIZE = 16
LEARNING_RATE = 0.00001
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


def get_fold_config(fold_id):
    """Get configuration for a specific LOPO fold.

    Args:
        fold_id: Fold number (0 to len(LOPO_PATIENTS)-1)

    Returns:
        dict with: fold_id, test_patient, train_patients, random_seed, output_prefix
    """
    n_folds = len(LOPO_PATIENTS)
    if fold_id < 0 or fold_id >= n_folds:
        raise ValueError(f"fold_id must be 0-{n_folds-1}, got {fold_id}")

    test_patient = LOPO_PATIENTS[fold_id]
    if LOPO_INCLUDE_OTHER_PATIENTS:
        train_patients = [p for p in LOPO_PATIENTS if p != test_patient]
    else:
        train_patients = []  # Single patient mode: no other patients in training

    return {
        "fold_id": fold_id,
        "test_patient": test_patient,
        "train_patients": train_patients,
        "random_seed": 42 + fold_id,
        "output_prefix": f"lopo_fold{fold_id:02d}",
    }
