"""Configuration for seizure prediction pipeline

Single Patient Training:
- Each patient is trained/tested on their own data
- Split strategy: Train on past seizures, test on future (last) seizure
"""

# =============================================================================
# Task Configuration
# =============================================================================

TASK_MODE = "prediction"  # 'prediction' (preictal vs interictal) or 'detection' (ictal vs interictal)
PREICTAL_WINDOW = 10 * 60  # 10 minutes before seizure
INTERICTAL_BUFFER = 30 * 60

# =============================================================================
# Dataset Configuration
# =============================================================================

BASE_PATH = "physionet.org/files/chbmit/1.0.0/"
ESTIMATED_FILE_DURATION = 3600  # 1 hour (fallback if file duration unavailable)

# Patients to include in processing
PATIENTS = [
    "chb01",
    "chb02",
    #"chb03",
    "chb04",
    "chb05",
    "chb06",
    "chb07",
    "chb08",
    "chb09",
    "chb10",
    "chb11",
    #"chb13",
    "chb14",
    "chb15",
    #"chb16",
    #"chb17",
    "chb18",
    "chb19",
    "chb20",
    "chb21",
    "chb22",
    "chb23",
]

# Current patient index to process (0 to len(PATIENTS)-1, or None for all)
PATIENT_INDEX = None

# Precomputed seizure counts (for reference)
from data_segmentation_helpers.seizure_counts import SEIZURE_COUNTS

# =============================================================================
# Sequence Configuration
# =============================================================================

SEGMENT_DURATION = 5  # seconds per segment
SEQUENCE_LENGTH = 5  # segments per sequence
SEQUENCE_STRIDE = 1  # segments between sequences

# =============================================================================
# Signal Processing
# =============================================================================

LOW_FREQ_HZ = 0.5
HIGH_FREQ_HZ = 128  # Extended to capture High Gamma/HFO
NOTCH_FREQ_HZ = 60

# Dual-Stream Configuration
PHASE_FREQ_BAND = (0.5, 12.0)  # Delta/Theta (for Phase)
AMP_FREQ_BAND = (20.0, 128.0)  # Gamma/HFO (for Amplitude)

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
LSTM_DROPOUT = 0.5  # Reduced from 0.6 back to 0.5

# =============================================================================
# Training Configuration
# =============================================================================

TRAINING_EPOCHS = 5
SEQUENCE_BATCH_SIZE = 16
LEARNING_RATE = 0.0001  # Increased from 1e-5 to 1e-4
WEIGHT_DECAY = 1e-4  # Reduced from 1e-3 back to 1e-4
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
