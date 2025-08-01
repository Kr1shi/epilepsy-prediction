"""Configuration for seizure prediction segmentation"""

# Segment configuration
SEGMENT_DURATION = 30  # seconds

# Seizure prediction parameters
PREICTAL_WINDOW = 30 * 60  # 30 minutes in seconds
SAFE_BUFFER = 30 * 60      # 30 minutes buffer

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

# STFT parameters
STFT_NPERSEG = 256      # Window length for STFT
STFT_NOVERLAP = 128     # Overlap between windows (50%)

# Artifact removal
ARTIFACT_THRESHOLD_STD = 4    # Standard deviations for artifact detection

# Bad channel detection  
BAD_CHANNEL_STD_THRESHOLD = 3   # Z-score threshold for bad channels
BAD_CHANNEL_FLAT_PERCENTILE = 5 # Percentile threshold for flat channels

# Power spectrogram settings
LOG_TRANSFORM_EPSILON = 1e-12   # Small value to avoid log(0)
APPLY_LOG_TRANSFORM = True      # Whether to apply log transform

TARGET_CHANNELS = [
    'C3-P3', 'C4-P4', 'CZ-PZ', 'F3-C3', 'F4-C4', 'F7-T7', 'F8-T8', 
    'FP1-F3', 'FP1-F7', 'FP2-F4', 'FP2-F8', 'FZ-CZ', 'P3-O1', 'P4-O2', 
    'P7-O1', 'P8-O2', 'T7-P7', 'T8-P8'
]

SKIP_CHANNEL_VALIDATION = True