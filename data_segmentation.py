"""Main execution - clean flow"""

import os
from config import *
from validation import validate_configuration, check_data_sufficiency, warn_if_needed
from segmentation import parse_summary_file, create_preictal_segments, create_interictal_segments

def create_prediction_segments():
    """Main function with clean flow"""
    
    
    # Parse data
    summary_path = f"{BASE_PATH}{PATIENT_ID}/{PATIENT_ID}-summary.txt"
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    seizures, all_files = parse_summary_file(summary_path)
    
    # Create segments
    preictal_segments = create_preictal_segments(seizures)
    interictal_segments = create_interictal_segments(all_files, seizures)
    
    # Validate results
    if REQUIRE_MINIMUM_SEGMENTS:
        check_data_sufficiency(len(preictal_segments), len(interictal_segments))
    
    # Optional warnings
    warn_if_needed(len(preictal_segments), len(interictal_segments))
    
    return preictal_segments, interictal_segments

def print_summary(preictal_segments, interictal_segments):
    """Print clean summary"""
    print(f"Patient: {PATIENT_ID}")
    print(f"Preictal segments: {len(preictal_segments)}")
    print(f"Interictal segments: {len(interictal_segments)}")
    print(f"Segment duration: {SEGMENT_DURATION}s")
    print(f"Preictal window: {PREICTAL_WINDOW//60}min")

if __name__ == "__main__":
    try:
        preictal_segs, interictal_segs = create_prediction_segments()
        print_summary(preictal_segs, interictal_segs)
        print("✅ Segmentation completed successfully!")
        print(preictal_segs, interictal_segs) 
    except Exception as e:
        print(f"❌ Error: {e}")