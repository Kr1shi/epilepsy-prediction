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

# Single-patient experiment configuration
SINGLE_PATIENT_MODE = True
SINGLE_PATIENT_ID = "chb06"

# Leave-One-Out Cross-Validation configuration (only supported mode)
LOOCV_MODE = True  # LOOCV is the only supported mode
LOOCV_FOLD_ID = 1   # Fold ID (0-9 for chb06) - test seizure for this fold
LOOCV_TOTAL_SEIZURES = 10  # Total number of seizures for the patient

# Random seed for reproducibility (fold-specific)
SINGLE_PATIENT_RANDOM_SEED = 42 + LOOCV_FOLD_ID

# Naming helper for outputs/datasets
if SINGLE_PATIENT_MODE:
    OUTPUT_PREFIX = f"{SINGLE_PATIENT_ID}_fold{LOOCV_FOLD_ID}"
else:
    OUTPUT_PREFIX = "all_patients"

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
