"""Configuration for seizure prediction segmentation"""

# Segment configuration
SEGMENT_DURATION = 5  # seconds

# Task mode configuration
TASK_MODE = 'detection'  # Options: 'prediction' (preictal vs interictal) or 'detection' (ictal vs interictal)

# Seizure prediction parameters
PREICTAL_WINDOW = 10 * 60       # 10 minutes in seconds - sequences within this window before seizure are preictal
INTERICTAL_BUFFER = 120 * 60     # 120 minutes (2 hours) - interictal sequences must be at least this far from any seizure

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
SINGLE_PATIENT_SEIZURE_SPLITS = {
    "train": [0, 1, 2, 3, 4, 5],
    "val": [6, 7],
    "test": [8, 9],
}
SINGLE_PATIENT_INTERICTAL_SPLIT = {
    "train": 0.6,
    "val": 0.2,
    "test": 0.2,
}
SINGLE_PATIENT_RANDOM_SEED = 42

# Naming helper for outputs/datasets
OUTPUT_PREFIX = SINGLE_PATIENT_ID if SINGLE_PATIENT_MODE else "all_patients"

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
TRAINING_EPOCHS = 15
BATCH_SIZE = 32
SEQUENCE_BATCH_SIZE = 4  # Small batch size for sequences (reduced from 12 to improve gradient estimates)
LEARNING_RATE = 0.0001  # Reduced from 0.001 to prevent divergence
NUM_WORKERS = 4  # DataLoader workers
WEIGHT_DECAY = 1e-4  # L2 regularization

# Preprocessing performance (M4 optimization)
PREPROCESSING_WORKERS = 12  # Number of parallel workers for preprocessing (adjust based on your CPU cores)
MNE_N_JOBS = 8  # Parallel jobs for MNE filtering operations
