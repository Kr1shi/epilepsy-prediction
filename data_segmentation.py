"""Sequence-based segmentation for CNN-LSTM model
Extracts sequences of consecutive 30-second segments for temporal modeling.

This module creates sequence metadata AND validates that files have all required
channels. Channel validation happens here (not in preprocessing) to fail fast
and avoid creating sequences that cannot be processed later.
"""
import os
import json
import mne
import random
from data_segmentation_helpers.config import *
from data_segmentation_helpers.segmentation import parse_summary_file
from data_segmentation_helpers.channel_validation import (
    validate_patient_files,
    get_validation_summary
)

def get_file_duration(edf_path):
    """Get actual duration of EDF file"""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        duration = raw.times[-1]  # Last timestamp
        return duration
    except Exception as e:
        if VERBOSE_WARNINGS:
            print(f"Warning: Could not read {edf_path}: {e}")
        return ESTIMATED_FILE_DURATION

def create_sequences_from_file(patient_id, filename, seizures_in_file, file_duration):
    """Create sequences of consecutive segments from a single file

    Args:
        patient_id: Patient identifier (e.g., 'chb01')
        filename: EDF filename (e.g., 'chb01_03.edf')
        seizures_in_file: List of seizure dicts for this file
        file_duration: Duration of the file in seconds

    Returns:
        List of sequence dictionaries
    """
    sequences = []

    # Calculate sequence parameters
    sequence_duration = SEGMENT_DURATION * SEQUENCE_LENGTH  # e.g., 30s * 10 = 300s
    stride_duration = SEGMENT_DURATION * SEQUENCE_STRIDE    # e.g., 30s * 5 = 150s

    # Calculate how many sequences we can extract
    if file_duration < sequence_duration:
        return sequences  # File too short for even one sequence

    # Generate sequence start times using sliding window
    sequence_start = 0
    while sequence_start + sequence_duration <= file_duration:
        sequence_end = sequence_start + sequence_duration

        # Generate segment start times within this sequence
        segment_starts = [sequence_start + (i * SEGMENT_DURATION) for i in range(SEQUENCE_LENGTH)]

        # Determine sequence label based on LAST segment
        last_segment_start = segment_starts[-1]
        last_segment_end = last_segment_start + SEGMENT_DURATION

        # Check if last segment is preictal or interictal
        sequence_type = 'interictal'
        time_to_seizure = None

        for seizure in seizures_in_file:
            seizure_start = seizure['start_sec']

            # Check if last segment falls in preictal window
            preictal_window_start = max(0, seizure_start - PREICTAL_WINDOW)

            if preictal_window_start <= last_segment_start < seizure_start:
                sequence_type = 'preictal'
                time_to_seizure = seizure_start - last_segment_end
                break

        # Create sequence metadata
        sequence = {
            'patient_id': patient_id,
            'file': filename,
            'sequence_start_sec': sequence_start,
            'sequence_end_sec': sequence_end,
            'sequence_duration_sec': sequence_duration,
            'segment_starts': segment_starts,
            'num_segments': SEQUENCE_LENGTH,
            'type': sequence_type
        }

        if sequence_type == 'preictal':
            sequence['time_to_seizure'] = time_to_seizure

        sequences.append(sequence)

        # Move to next sequence with stride
        sequence_start += stride_duration

    return sequences

def create_sequences_single_patient(patient_id):
    """Create sequences for a single patient with channel validation

    Returns:
        Tuple of (sequences, validation_results)
    """

    # Parse summary file
    summary_path = f"{BASE_PATH}{patient_id}/{patient_id}-summary.txt"
    if not os.path.exists(summary_path):
        print(f"Warning: Summary file not found for {patient_id}: {summary_path}")
        return [], None

    try:
        seizures, all_files = parse_summary_file(summary_path)

        # Validate channels for all files
        validation_results = validate_patient_files(patient_id, all_files, BASE_PATH)

        # Only process files with valid channels
        valid_files = set(validation_results['valid_file_list'])

        # Report validation results
        if validation_results['invalid_files'] > 0:
            print(f"  Channel validation: {validation_results['valid_files']}/{validation_results['total_files']} files valid")
            if VERBOSE_WARNINGS:
                for invalid_info in validation_results['invalid_file_info']:
                    print(f"    Skipping {invalid_info['file']}: missing {invalid_info['missing_channels']}")

        all_sequences = []

        # Process each VALID file
        for filename in all_files:
            # Skip files that don't have all required channels
            if filename not in valid_files:
                continue

            edf_path = f"{BASE_PATH}{patient_id}/{filename}"

            if not os.path.exists(edf_path):
                if VERBOSE_WARNINGS:
                    print(f"Warning: File not found: {edf_path}")
                continue

            # Get actual file duration
            file_duration = get_file_duration(edf_path)

            # Get seizures that occur in this file
            seizures_in_file = [s for s in seizures if s['file'] == filename]

            # Create sequences from this file
            sequences = create_sequences_from_file(patient_id, filename, seizures_in_file, file_duration)
            all_sequences.extend(sequences)

        # Count preictal and interictal
        preictal_count = len([s for s in all_sequences if s['type'] == 'preictal'])
        interictal_count = len([s for s in all_sequences if s['type'] == 'interictal'])

        print(f"Patient {patient_id}: {len(all_sequences)} sequences ({preictal_count} preictal, {interictal_count} interictal)")

        return all_sequences, validation_results

    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        import traceback
        traceback.print_exc()
        return [], None

def create_sequences_all_patients():
    """Process all patients and combine sequences

    Returns:
        Tuple of (all_sequences, all_validation_results)
    """

    all_sequences = []
    all_validation_results = []

    # Process patients chb01 to chb24, skipping chb12
    for i in range(1, 25):
        if i == 12:  # Skip patient 12
            continue

        patient_id = f"chb{i:02d}"  # Format as chb01, chb02, etc.

        print(f"\nProcessing {patient_id}...")
        sequences, validation_results = create_sequences_single_patient(patient_id)
        all_sequences.extend(sequences)

        if validation_results:
            all_validation_results.append(validation_results)

    return all_sequences, all_validation_results

def balance_sequences_per_patient(sequences, random_seed=42):
    """Balance sequences per patient by downsampling interictal to match preictal

    For each patient, randomly removes interictal sequences until the count
    matches the preictal count. This ensures 1:1 balance within each patient.

    Args:
        sequences: List of sequence dictionaries
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (balanced_sequences, balancing_stats)
    """
    random.seed(random_seed)

    # Group sequences by patient
    patient_sequences = {}
    for seq in sequences:
        patient_id = seq['patient_id']
        if patient_id not in patient_sequences:
            patient_sequences[patient_id] = {'preictal': [], 'interictal': []}
        patient_sequences[patient_id][seq['type']].append(seq)

    balanced_sequences = []
    balancing_stats = {
        'patients': {},
        'total_original': len(sequences),
        'total_balanced': 0,
        'total_removed': 0,
        'patients_processed': 0
    }

    # Balance each patient
    for patient_id in sorted(patient_sequences.keys()):
        preictal = patient_sequences[patient_id]['preictal']
        interictal = patient_sequences[patient_id]['interictal']

        n_preictal = len(preictal)
        n_interictal = len(interictal)

        # Keep all preictal sequences
        balanced_sequences.extend(preictal)

        # Randomly sample interictal to match preictal count
        if n_interictal > n_preictal:
            # Downsample interictal
            sampled_interictal = random.sample(interictal, n_preictal)
            balanced_sequences.extend(sampled_interictal)
            removed = n_interictal - n_preictal
        else:
            # Keep all interictal if fewer than preictal
            balanced_sequences.extend(interictal)
            removed = 0

        # Track statistics
        balancing_stats['patients'][patient_id] = {
            'preictal_original': n_preictal,
            'interictal_original': n_interictal,
            'interictal_kept': min(n_interictal, n_preictal),
            'interictal_removed': removed,
            'total_balanced': n_preictal + min(n_interictal, n_preictal)
        }
        balancing_stats['total_removed'] += removed
        balancing_stats['patients_processed'] += 1

    balancing_stats['total_balanced'] = len(balanced_sequences)

    return balanced_sequences, balancing_stats

def save_sequences_to_file(sequences, validation_results, output_file="all_patients_sequences.json", balancing_stats=None):
    """Save all sequences to a single file with validation information

    Args:
        sequences: List of sequence dictionaries
        validation_results: List of validation result dictionaries from each patient
        output_file: Output JSON filename
    """

    # Separate by type
    preictal_sequences = [s for s in sequences if s['type'] == 'preictal']
    interictal_sequences = [s for s in sequences if s['type'] == 'interictal']

    # Get validation summary
    validation_summary = get_validation_summary(validation_results) if validation_results else {}

    data = {
        'sequences': sequences,
        'preictal_sequences': preictal_sequences,
        'interictal_sequences': interictal_sequences,
        'summary': {
            'total_sequences': len(sequences),
            'total_preictal': len(preictal_sequences),
            'total_interictal': len(interictal_sequences),
            'patients_processed': len(set([s['patient_id'] for s in sequences])),
            'segment_duration': SEGMENT_DURATION,
            'sequence_length': SEQUENCE_LENGTH,
            'sequence_stride': SEQUENCE_STRIDE,
            'sequence_total_duration': SEGMENT_DURATION * SEQUENCE_LENGTH,
            'preictal_window': PREICTAL_WINDOW,
            'class_balance': len(preictal_sequences) / len(interictal_sequences) if interictal_sequences else 0
        },
        'validation_info': validation_summary
    }

    # Add balancing info if provided
    if balancing_stats:
        data['balancing_info'] = balancing_stats

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSequences saved to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

def print_summary(sequences):
    """Print comprehensive summary"""

    preictal_sequences = [s for s in sequences if s['type'] == 'preictal']
    interictal_sequences = [s for s in sequences if s['type'] == 'interictal']

    # Overall summary
    print(f"\n=== OVERALL SUMMARY ===")
    print(f"Total sequences: {len(sequences)}")
    print(f"  - Preictal sequences: {len(preictal_sequences)}")
    print(f"  - Interictal sequences: {len(interictal_sequences)}")
    print(f"Sequence configuration:")
    print(f"  - Segments per sequence: {SEQUENCE_LENGTH}")
    print(f"  - Sequence duration: {SEGMENT_DURATION * SEQUENCE_LENGTH}s ({SEGMENT_DURATION * SEQUENCE_LENGTH / 60:.1f} min)")
    print(f"  - Sequence stride: {SEQUENCE_STRIDE} segments ({SEGMENT_DURATION * SEQUENCE_STRIDE}s)")
    print(f"  - Preictal window: {PREICTAL_WINDOW//60} min")

    # Per-patient breakdown
    patients = set([s['patient_id'] for s in sequences])
    print(f"\n=== PER-PATIENT BREAKDOWN ===")
    print(f"Patients processed: {len(patients)}")

    for patient in sorted(patients):
        patient_seqs = [s for s in sequences if s['patient_id'] == patient]
        preictal_count = len([s for s in patient_seqs if s['type'] == 'preictal'])
        interictal_count = len([s for s in patient_seqs if s['type'] == 'interictal'])
        print(f"{patient}: {len(patient_seqs)} total ({preictal_count} preictal, {interictal_count} interictal)")

    # Statistics
    print(f"\n=== STATISTICS ===")
    if preictal_sequences:
        avg_time_to_seizure = sum([s.get('time_to_seizure', 0) for s in preictal_sequences]) / len(preictal_sequences)
        print(f"Average time to seizure (preictal): {avg_time_to_seizure:.1f}s ({avg_time_to_seizure/60:.1f} min)")

    print(f"Class balance: {len(preictal_sequences)}/{len(interictal_sequences)} = {len(preictal_sequences)/len(interictal_sequences) if interictal_sequences else 0:.2f}")

if __name__ == "__main__":
    try:
        print("="*60)
        print("SEQUENCE-BASED SEGMENTATION FOR CNN-LSTM")
        print("="*60)
        print(f"Configuration:")
        print(f"  - Sequence length: {SEQUENCE_LENGTH} segments")
        print(f"  - Sequence duration: {SEGMENT_DURATION * SEQUENCE_LENGTH}s ({SEGMENT_DURATION * SEQUENCE_LENGTH / 60:.1f} min)")
        print(f"  - Stride: {SEQUENCE_STRIDE} segments (50% overlap)")
        print(f"  - Segment duration: {SEGMENT_DURATION}s")
        print(f"  - Channel validation: {'ENABLED' if not SKIP_CHANNEL_VALIDATION else 'DISABLED'}")
        print(f"  - Target channels: {len(TARGET_CHANNELS)} channels")
        print("="*60)

        print("\nProcessing all patients...")
        all_sequences, all_validation_results = create_sequences_all_patients()

        if not all_sequences:
            print("❌ No sequences generated!")
        else:
            # Print unbalanced summary
            print("\n" + "="*60)
            print("BEFORE BALANCING")
            print("="*60)
            print_summary(all_sequences)

            # Balance sequences per patient
            print("\n" + "="*60)
            print("BALANCING SEQUENCES PER PATIENT")
            print("="*60)
            print("Downsampling interictal sequences to match preictal counts per patient...")

            balanced_sequences, balancing_stats = balance_sequences_per_patient(all_sequences)

            print(f"\nBalancing complete:")
            print(f"  - Original sequences: {balancing_stats['total_original']}")
            print(f"  - Balanced sequences: {balancing_stats['total_balanced']}")
            print(f"  - Removed sequences: {balancing_stats['total_removed']}")
            print(f"  - Reduction: {balancing_stats['total_removed']/balancing_stats['total_original']*100:.1f}%")

            # Print balanced summary
            print("\n" + "="*60)
            print("AFTER BALANCING")
            print("="*60)
            print_summary(balanced_sequences)

            # Print per-patient balancing details
            print("\n=== PER-PATIENT BALANCING DETAILS ===")
            for patient_id in sorted(balancing_stats['patients'].keys()):
                stats = balancing_stats['patients'][patient_id]
                print(f"{patient_id}: {stats['preictal_original']} preictal, "
                      f"{stats['interictal_original']} → {stats['interictal_kept']} interictal "
                      f"(removed {stats['interictal_removed']})")

            # Save balanced sequences to file
            save_sequences_to_file(balanced_sequences, all_validation_results,
                                  output_file="all_patients_sequences.json",
                                  balancing_stats=balancing_stats)

            # Print validation summary
            if all_validation_results:
                validation_summary = get_validation_summary(all_validation_results)
                print(f"\n=== CHANNEL VALIDATION SUMMARY ===")
                print(f"Validation enabled: {validation_summary['validation_enabled']}")
                print(f"Files checked: {validation_summary['total_files_checked']}")
                print(f"Files with valid channels: {validation_summary['files_with_valid_channels']}")
                print(f"Files with invalid channels: {validation_summary['files_with_invalid_channels']}")

                if validation_summary['files_with_invalid_channels'] > 0:
                    print(f"\nMost common missing channels:")
                    for channel, count in list(validation_summary['missing_channel_frequency'].items())[:5]:
                        print(f"  - {channel}: missing in {count} files")

            print("\n✅ Sequence segmentation completed successfully!")
            print(f"✅ Balanced sequences saved to all_patients_sequences.json")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
