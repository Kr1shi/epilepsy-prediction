"""Configuration for seizure prediction segmentation"""

# Segment configuration
SEGMENT_DURATION = 5  # seconds

# Task mode configuration
TASK_MODE = 'prediction'  
# Seizure prediction parameters
PREICTAL_WINDOW = 10 * 60       # 10 minutes 
INTERICTAL_BUFFER = 120 * 60     # 120 minutes 

# File assumptions
ESTIMATED_FILE_DURATION = 3600  # 1 hour

# Dataset configuration
BASE_PATH = "physionet.org/files/chbmit/1.0.0/"

# Error handling (set to False for cleaner output)
VERBOSE_WARNINGS = False
REQUIRE_MINIMUM_SEGMENTS = True
MIN_SEGMENTS_THRESHOLD = 1  

LOW_FREQ_HZ = 0.5
HIGH_FREQ_HZ = 50
NOTCH_FREQ_HZ = 60

# Single-patient configuration (LOOCV operates on single patients only)
SINGLE_PATIENT_ID = "chb11"

# Leave-One-Out Cross-Validation configuration (only supported mode)
LOOCV_MODE = True  # LOOCV is the only supported mode
LOOCV_FOLD_ID = None  # Set to fold number or None to process all folds

# Load precomputed seizure counts
from data_segmentation_helpers.seizure_counts import SEIZURE_COUNTS

if SINGLE_PATIENT_ID not in SEIZURE_COUNTS:
    raise ValueError(
        f"Patient {SINGLE_PATIENT_ID} not found in seizure_counts\n"
        f"Available patients: {sorted(SEIZURE_COUNTS.keys())}"
    )
LOOCV_TOTAL_SEIZURES = SEIZURE_COUNTS[SINGLE_PATIENT_ID]

def get_fold_config(fold_id):
    """Get fold-specific configuration values

    Args:
        fold_id: Fold ID (0 to LOOCV_TOTAL_SEIZURES-1)

    Returns:
        Dictionary with fold-specific config:
        - fold_id: The fold number
        - random_seed: Fold-specific random seed (42 + fold_id)
        - output_prefix: Fold-specific output prefix
    """
    return {
        'fold_id': fold_id,
        'random_seed': 42 + fold_id,
        'output_prefix': f"{SINGLE_PATIENT_ID}_fold{fold_id}"
    }

# Compute values only when LOOCV_FOLD_ID is set
if LOOCV_FOLD_ID is not None:
    SINGLE_PATIENT_RANDOM_SEED = 42 + LOOCV_FOLD_ID
    OUTPUT_PREFIX = f"{SINGLE_PATIENT_ID}_fold{LOOCV_FOLD_ID}"
else:
    # Placeholder values when processing all folds
    SINGLE_PATIENT_RANDOM_SEED = None
    OUTPUT_PREFIX = None

# STFT parameters
STFT_NPERSEG = 256      # Window length for STFT
STFT_NOVERLAP = 128     # Overlap between windows (50%)

# Artifact removal
ARTIFACT_THRESHOLD_STD = 4    # Standard deviations for artifact detection (Use 4 if you want artifact removal)

# Power spectrogram settings
LOG_TRANSFORM_EPSILON = 1e-12   # Small value to avoid log(0)
APPLY_LOG_TRANSFORM = True      # Whether to apply log transform

TARGET_CHANNELS = [
    'C3-P3', 'C4-P4', 'CZ-PZ', 'F3-C3', 'F4-C4', 'F7-T7', 'F8-T8', 
    'FP1-F3', 'FP1-F7', 'FP2-F4', 'FP2-F8', 'FZ-CZ', 'P3-O1', 'P4-O2', 
    'P7-O1', 'P8-O2', 'T7-P7', 'T8-P8'
]

SKIP_CHANNEL_VALIDATION = False

# Sequence configuration for CNN-LSTM
SEQUENCE_LENGTH = 30        
SEQUENCE_STRIDE = 5        

# LSTM configuration
LSTM_HIDDEN_DIM = 512      # LSTM hidden state dimension (increased for deeper CNN)
LSTM_NUM_LAYERS = 3        # Number of stacked LSTM layers (increased depth)
LSTM_DROPOUT = 0.5         # Dropout between LSTM layers (increased regularization)

# Training configuration
TRAINING_EPOCHS = 5
SEQUENCE_BATCH_SIZE = 16  # Batch size for sequences
LEARNING_RATE = 0.00001
NUM_WORKERS = 4  # DataLoader workers
WEIGHT_DECAY = 1e-4  # L2 regularization

# Preprocessing performance (M4 optimization)
PREPROCESSING_WORKERS = 10  # Number of parallel workers for preprocessing (adjust based on your CPU cores)
MNE_N_JOBS = 8  # Parallel jobs for MNE filtering operations
