"""Validation and error checking for segmentation"""

from config import *

def validate_configuration():
    """Validate configuration and return issues"""
    issues = []
    
    if PREICTAL_WINDOW > ESTIMATED_FILE_DURATION:
        issues.append(f"Preictal window ({PREICTAL_WINDOW//60}min) > file duration")
    
    if SEGMENT_DURATION >= PREICTAL_WINDOW:
        issues.append(f"Segment duration >= preictal window")
    
    max_possible = PREICTAL_WINDOW // SEGMENT_DURATION
    if NUM_PREICTAL_SEGMENTS > max_possible:
        issues.append(f"Too many preictal segments requested ({NUM_PREICTAL_SEGMENTS} > {max_possible})")
    
    return issues

def check_data_sufficiency(preictal_count, interictal_count):
    """Check if we have sufficient data"""
    min_preictal = int(NUM_PREICTAL_SEGMENTS * MIN_SEGMENTS_THRESHOLD)
    min_interictal = int(NUM_INTERICTAL_SEGMENTS * MIN_SEGMENTS_THRESHOLD)
    
    if preictal_count < min_preictal:
        raise ValueError(f"Insufficient preictal data: {preictal_count} < {min_preictal}")
    
    if interictal_count < min_interictal:
        raise ValueError(f"Insufficient interictal data: {interictal_count} < {min_interictal}")

def warn_if_needed(preictal_count, interictal_count):
    """Print warnings only if verbose mode is on"""
    if not VERBOSE_WARNINGS:
        return
        
    if preictal_count < NUM_PREICTAL_SEGMENTS:
        print(f"Warning: Only {preictal_count}/{NUM_PREICTAL_SEGMENTS} preictal segments")
    
    if interictal_count < NUM_INTERICTAL_SEGMENTS:
        print(f"Warning: Only {interictal_count}/{NUM_INTERICTAL_SEGMENTS} interictal segments")