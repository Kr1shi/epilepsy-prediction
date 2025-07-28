import os
import glob
import pandas as pd
import pyedflib
import numpy as np

def parse_summary_file(summary_path):
    """Parse chbXX-summary.txt to extract seizure times - handles multiple formats"""
    seizures = []
    
    with open(summary_path, 'r') as f:
        lines = f.readlines()
    
    current_file = None
    current_seizures = 0
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for file names
        if line.startswith('File Name:'):
            current_file = line.split(':', 1)[1].strip()
            
        # Look for number of seizures
        elif 'Number of Seizures in File:' in line:
            try:
                current_seizures = int(line.split(':')[-1].strip())
                
                # If there are seizures, read them
                if current_seizures > 0:
                    for j in range(current_seizures):
                        # Look for start time - handle both formats
                        i += 1
                        if i < len(lines):
                            start_line = lines[i].strip()
                            # Handle "Seizure N Start Time:" or "Seizure Start Time:"
                            if 'Start Time:' in start_line:
                                start_time = int(start_line.split(':')[-1].strip().split(' ')[0])
                            else:
                                print(f"WARNING: Expected start time but got: {start_line}")
                                continue
                        
                        # Look for end time
                        i += 1
                        if i < len(lines):
                            end_line = lines[i].strip()
                            if 'End Time:' in end_line:
                                end_time = int(end_line.split(':')[-1].strip().split(' ')[0])
                            else:
                                print(f"WARNING: Expected end time but got: {end_line}")
                                continue
                            
                            seizures.append({
                                'file': current_file,
                                'start_sec': start_time,
                                'end_sec': end_time,
                                'duration_sec': end_time - start_time
                            })
                        
            except ValueError as e:
                print(f"Error parsing seizure count from: '{line}' - {e}")
        
        i += 1
    
    return seizures

def get_channel_info(summary_path):
    """Extract channel information and changes from summary file"""
    channel_configs = []
    
    with open(summary_path, 'r') as f:
        lines = f.readlines()
    
    current_channels = []
    recording_channels = False
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('Channels in EDF Files:') or line.startswith('Channels changed:'):
            recording_channels = True
            if current_channels:  # Save previous config
                channel_configs.append(current_channels.copy())
            current_channels = []
            continue
            
        if recording_channels and line.startswith('Channel '):
            try:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    channel_name = parts[1].strip()
                    current_channels.append(channel_name)
            except:
                continue
                
        elif recording_channels and (line.startswith('File Name:') or line == ''):
            recording_channels = False
    
    # Add final config
    if current_channels:
        channel_configs.append(current_channels)
    
    return channel_configs

def get_common_channels_per_patient(patient_folder):
    """Find channels that are present in ALL EDF files for a patient"""
    
    edf_files = glob.glob(f"{patient_folder}/*.edf")
    if not edf_files:
        return [], []
    
    all_channels_per_file = []
    file_channel_info = []
    
    for edf_file in edf_files:
        try:
            f = pyedflib.EdfReader(edf_file)
            channels = f.getSignalLabels()
            f.close()
            
            # Remove dummy channels (marked as "-")
            valid_channels = [ch for ch in channels if ch != "-"]
            all_channels_per_file.append(set(valid_channels))
            
            file_channel_info.append({
                'file': os.path.basename(edf_file),
                'total_channels': len(channels),
                'valid_channels': len(valid_channels),
                'channels': valid_channels
            })
            
        except Exception as e:
            print(f"Error reading {edf_file}: {e}")
            continue
    
    # Find intersection - channels present in ALL files
    if all_channels_per_file:
        common_channels = set.intersection(*all_channels_per_file)
        common_channels = sorted(list(common_channels))
    else:
        common_channels = []
    
    return common_channels, file_channel_info

def analyze_patient_folder(patient_folder):
    """Analyze one patient folder for time distributions"""
    
    # Find summary file
    summary_files = glob.glob(f"{patient_folder}/*-summary.txt")
    if not summary_files:
        print(f"No summary file found in {patient_folder}")
        return None
    
    # Parse seizures
    seizures = parse_summary_file(summary_files[0])
    
    # Get all EDF files and calculate total duration
    edf_files = glob.glob(f"{patient_folder}/*.edf")
    total_duration_sec = 0
    
    for edf_file in edf_files:
        try:
            f = pyedflib.EdfReader(edf_file)
            duration = f.getNSamples()[0] / f.getSampleFrequency(0)
            total_duration_sec += duration
            f.close()
        except Exception as e:
            print(f"Error reading {edf_file}: {e}")
            continue
    
    # Calculate seizure statistics
    num_seizures = len(seizures)
    ictal_duration = sum([s['duration_sec'] for s in seizures])
    
    # Calculate preictal duration (30 min before each seizure)
    preictal_duration_per_seizure = 30 * 60  # 30 minutes
    actual_preictal_duration = 0
    for seizure in seizures:
        if seizure['start_sec'] >= preictal_duration_per_seizure:
            actual_preictal_duration += preictal_duration_per_seizure
        else:
            actual_preictal_duration += seizure['start_sec']
    
    # Interictal is everything else
    interictal_duration = total_duration_sec - ictal_duration - actual_preictal_duration
    
    # Convert to hours and percentages
    total_hours = total_duration_sec / 3600
    ictal_hours = ictal_duration / 3600
    preictal_hours = actual_preictal_duration / 3600
    interictal_hours = interictal_duration / 3600
    
    ictal_pct = (ictal_duration / total_duration_sec) * 100 if total_duration_sec > 0 else 0
    preictal_pct = (actual_preictal_duration / total_duration_sec) * 100 if total_duration_sec > 0 else 0
    interictal_pct = (interictal_duration / total_duration_sec) * 100 if total_duration_sec > 0 else 0
    
    return {
        'patient': os.path.basename(patient_folder),
        'num_seizures': num_seizures,
        'total_hours': total_hours,
        'ictal_hours': ictal_hours,
        'preictal_hours': preictal_hours,
        'interictal_hours': interictal_hours,
        'ictal_pct': ictal_pct,
        'preictal_pct': preictal_pct,
        'interictal_pct': interictal_pct,
        'seizures': seizures,
        'num_files': len(edf_files)
    }

def analyze_patient_folder_with_channels(patient_folder):
    """Analyze patient folder including channel consistency"""
    
    # Get basic analysis
    basic_analysis = analyze_patient_folder(patient_folder)
    if not basic_analysis:
        return None
    
    # Add channel analysis
    common_channels, file_channel_info = get_common_channels_per_patient(patient_folder)
    
    # Channel statistics
    all_unique_channels = set()
    channel_counts = {}
    
    for file_info in file_channel_info:
        for ch in file_info['channels']:
            all_unique_channels.add(ch)
            channel_counts[ch] = channel_counts.get(ch, 0) + 1
    
    total_files = len(file_channel_info)
    
    # Channels that appear in some but not all files
    inconsistent_channels = []
    for ch in all_unique_channels:
        if channel_counts[ch] < total_files:
            inconsistent_channels.append({
                'channel': ch,
                'appears_in': channel_counts[ch],
                'missing_from': total_files - channel_counts[ch]
            })
    
    # Add to basic analysis
    basic_analysis.update({
        'common_channels': common_channels,
        'num_common_channels': len(common_channels),
        'total_unique_channels': len(all_unique_channels),
        'inconsistent_channels': inconsistent_channels,
        'file_channel_info': file_channel_info,
        'channel_consistency': len(common_channels) / len(all_unique_channels) if all_unique_channels else 0
    })
    
    return basic_analysis

num_files = 25 

def check_summary_files(base_path="physionet.org/files/chbmit/1.0.0/"):
    """Check which summary files exist and their sizes"""
    
    print("CHECKING SUMMARY FILES:")
    print("="*50)
    
    for i in range(1, num_files):
        patient_folder = f"{base_path}chb{i:02d}"
        
        if os.path.exists(patient_folder):
            summary_files = glob.glob(f"{patient_folder}/*-summary.txt")
            if summary_files:
                summary_file = summary_files[0]
                file_size = os.path.getsize(summary_file)
                with open(summary_file, 'r') as f:
                    line_count = len(f.readlines())
                print(f"chb{i:02d}: {os.path.basename(summary_file)} ({file_size} bytes, {line_count} lines)")
            else:
                print(f"chb{i:02d}: NO SUMMARY FILE FOUND")
        else:
            print(f"chb{i:02d}: FOLDER NOT FOUND")

def analyze_all_patients_with_channels(base_path="physionet.org/files/chbmit/1.0.0/"):
    """Analyze all patients including channel consistency"""
    
    results = []
    
    for i in range(1, num_files):
        patient_folder = f"{base_path}chb{i:02d}"
        
        if os.path.exists(patient_folder):
            print(f"Analyzing {patient_folder}...")
            result = analyze_patient_folder_with_channels(patient_folder)
            if result:
                results.append(result)
        else:
            print(f"Folder not found: {patient_folder}")
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    # Display summary table
    print("\n" + "="*120)
    print("CHB-MIT DATASET ANALYSIS WITH CHANNEL CONSISTENCY")
    print("="*120)
    
    summary_cols = ['patient', 'num_seizures', 'total_hours', 'num_common_channels', 
                   'total_unique_channels', 'channel_consistency', 'ictal_pct', 'preictal_pct']
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df[summary_cols].round(3).to_string(index=False))
    
    # Channel consistency details
    print("\n" + "="*100)
    print("CHANNEL CONSISTENCY DETAILS")
    print("="*100)
    
    for _, row in df.iterrows():
        print(f"\n{row['patient']}:")
        print(f"  Common channels ({len(row['common_channels'])}): {row['common_channels']}")
        
        if row['inconsistent_channels']:
            print(f"  Inconsistent channels ({len(row['inconsistent_channels'])}):")
            for ch_info in row['inconsistent_channels']:
                print(f"    {ch_info['channel']}: appears in {ch_info['appears_in']}/{len(row['file_channel_info'])} files")
    
    # Overall statistics
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    print(f"Total patients: {len(df)}")
    print(f"Total seizures: {df['num_seizures'].sum()}")
    print(f"Average common channels per patient: {df['num_common_channels'].mean():.1f}")
    print(f"Average channel consistency: {df['channel_consistency'].mean():.3f}")
    
    # Find channels common across ALL patients
    if len(df) > 0:
        all_patient_common_channels = set(df.iloc[0]['common_channels'])
        for _, row in df.iterrows():
            all_patient_common_channels = all_patient_common_channels.intersection(set(row['common_channels']))
        
        print(f"\nChannels common across ALL patients ({len(all_patient_common_channels)}):")
        print(f"  {sorted(list(all_patient_common_channels))}")
    
    # Show seizure details for each patient
    print("\n" + "="*80)
    print("SEIZURE DETAILS BY PATIENT")
    print("="*80)
    for _, row in df.iterrows():
        print(f"\n{row['patient']}: {row['num_seizures']} seizures")
        for seizure in row['seizures']:
            print(f"  {seizure['file']}: {seizure['start_sec']}s - {seizure['end_sec']}s ({seizure['duration_sec']}s)")
    
    return df

if __name__ == "__main__":
    # The purpose of this code is to give us information collected in the chbmit dataset. Per case, it will provide us with the number of seizures, the common channels and the percentage of EEG recording time in preictal and ictal stage
    # First check if summary files exist
    check_summary_files()
    print("\n")
    
    # Run the complete analysis
    df_analysis = analyze_all_patients_with_channels()
    
    # Save results to CSV for later use
    df_analysis.to_csv('chb_mit_analysis.csv', index=False)
    print(f"\nAnalysis saved to 'chb_mit_analysis.csv'")