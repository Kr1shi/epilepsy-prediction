"""Configuration for seizure prediction segmentation"""

# Segment configuration
NUM_PREICTAL_SEGMENTS = 984
NUM_INTERICTAL_SEGMENTS = 984 
SEGMENT_DURATION = 10  # seconds

# Seizure prediction parameters
PREICTAL_WINDOW = 30 * 60  # 30 minutes in seconds
SAFE_BUFFER = 30 * 60      # 30 minutes buffer

# File assumptions
ESTIMATED_FILE_DURATION = 3600  # 1 hour
MAX_SEGMENTS_PER_FILE = 5

# Dataset configuration
BASE_PATH = "physionet.org/files/chbmit/1.0.0/"
PATIENT_ID = "chb01"

# Error handling (set to False for cleaner output)
VERBOSE_WARNINGS = False
REQUIRE_MINIMUM_SEGMENTS = True
MIN_SEGMENTS_THRESHOLD = 1  