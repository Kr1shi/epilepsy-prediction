"""Main execution - process all patients and combine segments"""
import os
import json
from data_segmentation_helpers.config import *
from data_segmentation_helpers.segmentation import parse_summary_file, create_preictal_segments, create_interictal_segments

def create_prediction_segments_single_patient(patient_id):
    """Create segments for a single patient"""
    
    # Parse data
    summary_path = f"{BASE_PATH}{patient_id}/{patient_id}-summary.txt"
    if not os.path.exists(summary_path):
        print(f"Warning: Summary file not found for {patient_id}: {summary_path}")
        return [], []
    
    try:
        seizures, all_files = parse_summary_file(summary_path)
        
        # Create segments
        preictal_segments = create_preictal_segments(seizures)
        interictal_segments = create_interictal_segments(all_files, seizures)
        
        # Add patient ID to each segment
        for segment in preictal_segments:
            segment['patient_id'] = patient_id
        for segment in interictal_segments:
            segment['patient_id'] = patient_id
        
        print(f"Patient {patient_id}: {len(preictal_segments)} preictal, {len(interictal_segments)} interictal")
        
        return preictal_segments, interictal_segments
        
    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        return [], []

def create_prediction_segments_all_patients():
    """Process all patients and combine segments"""
    
    all_preictal_segments = []
    all_interictal_segments = []
    
    # Process patients chb01 to chb24, skipping chb12
    for i in range(1, 25):
        if i == 12:  # Skip patient 12
            continue
            
        patient_id = f"chb{i:02d}"  # Format as chb01, chb02, etc.
        
        print(f"\nProcessing {patient_id}...")
        preictal_segs, interictal_segs = create_prediction_segments_single_patient(patient_id)
        
        all_preictal_segments.extend(preictal_segs)
        all_interictal_segments.extend(interictal_segs)
    
    return all_preictal_segments, all_interictal_segments

def save_segments_to_file(preictal_segments, interictal_segments, output_file="all_patients_segments.json"):
    """Save all segments to a single file"""
    
    data = {
        'preictal_segments': preictal_segments,
        'interictal_segments': interictal_segments,
        'summary': {
            'total_preictal': len(preictal_segments),
            'total_interictal': len(interictal_segments),
            'patients_processed': len(set([s['patient_id'] for s in preictal_segments + interictal_segments])),
            'segment_duration': SEGMENT_DURATION,
            'preictal_window': PREICTAL_WINDOW
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSegments saved to {output_file}")

def print_summary(preictal_segments, interictal_segments):
    """Print comprehensive summary"""
    
    # Overall summary
    print(f"\n=== OVERALL SUMMARY ===")
    print(f"Total preictal segments: {len(preictal_segments)}")
    print(f"Total interictal segments: {len(interictal_segments)}")
    print(f"Segment duration: {SEGMENT_DURATION}s")
    print(f"Preictal window: {PREICTAL_WINDOW//60}min")
    
    # Per-patient breakdown
    patients = set([s['patient_id'] for s in preictal_segments + interictal_segments])
    print(f"\n=== PER-PATIENT BREAKDOWN ===")
    print(f"Patients processed: {len(patients)}")
    
    for patient in sorted(patients):
        preictal_count = len([s for s in preictal_segments if s['patient_id'] == patient])
        interictal_count = len([s for s in interictal_segments if s['patient_id'] == patient])
        print(f"{patient}: {preictal_count} preictal, {interictal_count} interictal")

if __name__ == "__main__":
    try:
        print("Processing all patients...")
        all_preictal_segs, all_interictal_segs = create_prediction_segments_all_patients()
        
        # Save to file
        save_segments_to_file(all_preictal_segs, all_interictal_segs)
        
        # Print summary
        print_summary(all_preictal_segs, all_interictal_segs)
        
        print("\n✅ Multi-patient segmentation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")