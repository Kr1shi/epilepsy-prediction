"""Core segmentation logic - extract all possible segments"""

import os
from data_segmentation_helpers.config import *

def parse_summary_file(summary_path):
    """Parse summary file to extract seizures and files"""
    seizures = []
    all_files = []
    
    with open(summary_path, 'r') as f:
        lines = f.readlines()
    
    current_file = None
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('File Name:'):
            current_file = line.split(':', 1)[1].strip()
            all_files.append(current_file)
            
        elif 'Number of Seizures in File:' in line:
            num_seizures = int(line.split(':')[-1].strip())
            
            for j in range(num_seizures):
                i += 1
                start_time = int(lines[i].strip().split(':')[-1].strip().split(' ')[0])
                i += 1
                end_time = int(lines[i].strip().split(':')[-1].strip().split(' ')[0])
                
                seizures.append({
                    'file': current_file,
                    'start_sec': start_time,
                    'end_sec': end_time,
                    'duration_sec': end_time - start_time
                })
        
        i += 1
    
    return seizures, all_files

def create_preictal_segments(seizures):
    """Create ALL possible preictal segments from available time before seizures"""
    segments = []
    
    for seizure in seizures:
        # Use whatever time is available before the seizure
        if seizure['start_sec'] < SEGMENT_DURATION:
            continue  # Skip if we can't even fit one segment
            
        # Available preictal window: from start of file to seizure start
        # But prefer the last 30 minutes if available
        max_preictal_start = max(0, seizure['start_sec'] - PREICTAL_WINDOW)
        available_preictal_duration = seizure['start_sec'] - max_preictal_start
        
        max_segments = available_preictal_duration // SEGMENT_DURATION
        
        if max_segments == 0:
            continue
            
        # Extract ALL possible segments (no limit)
        for i in range(max_segments):
            # Distribute segments across the available preictal window
            segment_start = max_preictal_start + (i * (available_preictal_duration // max_segments))
            
            segments.append({
                'file': seizure['file'],
                'start_sec': segment_start,
                'end_sec': segment_start + SEGMENT_DURATION,
                'duration_sec': SEGMENT_DURATION,
                'type': 'preictal',
                'time_to_seizure': seizure['start_sec'] - (segment_start + SEGMENT_DURATION)
            })
    
    return segments

def create_interictal_segments(all_files, seizures):
    """Create ALL possible interictal segments from seizure-free files"""
    seizure_files = set([s['file'] for s in seizures])
    non_seizure_files = [f for f in all_files if f not in seizure_files]
    print("seizure files:", len(seizure_files), "non-seizure files:", len(non_seizure_files))
    
    segments = []
    
    for filename in non_seizure_files:
        # Calculate maximum segments that can fit in this file
        segments_from_file = ESTIMATED_FILE_DURATION // SEGMENT_DURATION
        
        for i in range(segments_from_file):
            segment_start = i * SEGMENT_DURATION 
            
            segments.append({
                'file': filename,
                'start_sec': segment_start,
                'end_sec': segment_start + SEGMENT_DURATION,
                'duration_sec': SEGMENT_DURATION,
                'type': 'interictal'
            })
    
    return segments