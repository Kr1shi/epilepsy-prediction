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

# Safety margin to ensure sufficient data for preprocessing (filter padding + STFT requirements)
# Preprocessing adds 5s padding for filter edge effects + margin for STFT windows
SAFETY_MARGIN = 6.0  # seconds

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

def identify_positive_regions(all_seizures_global):
    """Identify time regions where preictal/ictal sequences should be created.

    This determines where the sliding window should use overlapping sequences (83% overlap).
    Regions outside these bounds will use non-overlapping sequences (0% overlap) to prevent
    interictal data leakage between train/test splits.

    Args:
        all_seizures_global: List of seizures with global timeline coordinates

    Returns:
        List of (start_global, end_global) tuples marking regions where positive sequences occur
    """
    positive_regions = []

    for seizure in all_seizures_global:
        if TASK_MODE == 'prediction':
            # Prediction mode: preictal window before seizure
            seizure_start_global = seizure['start_sec_global']
            preictal_window_start_global = max(0, seizure_start_global - PREICTAL_WINDOW)
            # Region: from preictal window start to seizure start
            positive_regions.append((preictal_window_start_global, seizure_start_global))

        elif TASK_MODE == 'detection':
            # Detection mode: during seizure
            seizure_start_global = seizure['start_sec_global']
            seizure_end_global = seizure['end_sec_global']
            # Region: from seizure start to seizure end
            positive_regions.append((seizure_start_global, seizure_end_global))

    # Merge overlapping regions
    if not positive_regions:
        return []

    positive_regions.sort()
    merged = [positive_regions[0]]
    for start, end in positive_regions[1:]:
        if start <= merged[-1][1]:
            # Overlapping regions, merge them
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged

def create_sequences_from_file(patient_id, filename, all_seizures_global, file_duration, file_offset):
    """Create sequences of consecutive segments from a single file

    Args:
        patient_id: Patient identifier (e.g., 'chb01')
        filename: EDF filename (e.g., 'chb01_03.edf')
        all_seizures_global: List of ALL seizures with global timeline coordinates
        file_duration: Duration of the file in seconds
        file_offset: Global timeline offset for this file (accounts for gaps)

    Returns:
        List of sequence dictionaries
    """
    sequences = []
    skipped_boundary_sequences = 0  # Track sequences skipped due to insufficient boundary data

    # Calculate sequence parameters
    sequence_duration = SEGMENT_DURATION * SEQUENCE_LENGTH  # e.g., 5s * 30 = 150s
    overlapping_stride = SEGMENT_DURATION * SEQUENCE_STRIDE  # e.g., 5s * 5 = 25s (83% overlap for preictal/ictal)
    non_overlapping_stride = sequence_duration                # e.g., 150s (0% overlap for interictal)

    # Identify positive regions (preictal/ictal) to determine stride strategy
    positive_regions = identify_positive_regions(all_seizures_global)

    # Calculate how many sequences we can extract
    if file_duration < sequence_duration + (2 * SAFETY_MARGIN):
        return sequences  # File too short for even one sequence (including safety margins)

    # Helper function to check if a position is in a positive region
    def is_in_positive_region(start_global, end_global):
        """Check if sequence overlaps with any positive region"""
        for region_start, region_end in positive_regions:
            if start_global < region_end and end_global > region_start:
                return True
        return False

    # Generate sequence start times using adaptive sliding window
    sequence_start_local = SAFETY_MARGIN
    while sequence_start_local + sequence_duration + SAFETY_MARGIN <= file_duration:
        sequence_end_local = sequence_start_local + sequence_duration

        # Convert to global timeline
        sequence_start_global = file_offset + sequence_start_local
        sequence_end_global = file_offset + sequence_end_local

        # Generate segment start times within this sequence (local to file)
        segment_starts = [sequence_start_local + (i * SEGMENT_DURATION) for i in range(SEQUENCE_LENGTH)]

        # Validate all segments have sufficient data
        last_segment_end = segment_starts[-1] + SEGMENT_DURATION
        if last_segment_end + SAFETY_MARGIN > file_duration:
            # Skip this sequence - insufficient data for complete preprocessing
            skipped_boundary_sequences += 1
            # Use non-overlapping stride for interictal regions (safest assumption)
            sequence_start_local += non_overlapping_stride
            continue

        # Determine sequence label based on LAST segment
        last_segment_start_local = segment_starts[-1]
        last_segment_end_local = last_segment_start_local + SEGMENT_DURATION
        last_segment_start_global = file_offset + last_segment_start_local
        last_segment_end_global = file_offset + last_segment_end_local

        # Check if sequence is preictal/ictal, interictal, or in excluded buffer zone
        sequence_type = 'interictal'
        time_to_seizure = None
        in_excluded_zone = False
        matched_seizure_id = None

        # PASS 1: Check for positive class (preictal OR ictal depending on mode)
        if TASK_MODE == 'prediction':
            # Prediction mode: Check if preictal for ANY seizure (takes priority)
            for seizure in all_seizures_global:
                seizure_start_global = seizure['start_sec_global']

                # Check if last segment falls in preictal window (10 minutes before seizure)
                preictal_window_start_global = max(0, seizure_start_global - PREICTAL_WINDOW)

                if preictal_window_start_global <= last_segment_start_global < seizure_start_global:
                    sequence_type = 'preictal'
                    time_to_seizure = seizure_start_global - last_segment_end_global
                    matched_seizure_id = seizure.get('seizure_id')
                    break

        elif TASK_MODE == 'detection':
            # Detection mode: Check if ictal (overlaps with ANY seizure period)
            for seizure in all_seizures_global:
                seizure_start_global = seizure['start_sec_global']
                seizure_end_global = seizure['end_sec_global']

                # Check if ANY part of sequence overlaps with seizure period
                # Overlap occurs if: sequence_start < seizure_end AND sequence_end > seizure_start
                if sequence_start_global < seizure_end_global and sequence_end_global > seizure_start_global:
                    sequence_type = 'ictal'
                    matched_seizure_id = seizure.get('seizure_id')
                    # Store overlap information (could be useful for analysis)
                    overlap_start = max(sequence_start_global, seizure_start_global)
                    overlap_end = min(sequence_end_global, seizure_end_global)
                    break

        # PASS 2: Only check buffer zones if NOT positive class (preictal/ictal)
        # Buffer zones should only exclude interictal sequences, not override positive labels
        if sequence_type == 'interictal':
            for seizure in all_seizures_global:
                seizure_start_global = seizure['start_sec_global']
                seizure_end_global = seizure['end_sec_global']

                if TASK_MODE == 'prediction':
                    # Prediction mode: Exclude 50-min buffer before preictal window
                    preictal_window_start_global = max(0, seizure_start_global - PREICTAL_WINDOW)

                    # Buffer zone 1: [seizure_start - 60min, seizure_start - 10min) - before preictal window
                    buffer_before_start_global = max(0, seizure_start_global - INTERICTAL_BUFFER)
                    buffer_before_end_global = preictal_window_start_global

                    # Check if sequence overlaps with pre-preictal buffer
                    if not (sequence_end_global <= buffer_before_start_global or sequence_start_global >= buffer_before_end_global):
                        in_excluded_zone = True
                        break

                    # Buffer zone 2: [seizure_start, seizure_end + 60min] - during and after seizure
                    buffer_after_end_global = seizure_end_global + INTERICTAL_BUFFER

                    # Check if sequence overlaps with ictal + post-ictal buffer
                    if not (sequence_end_global <= seizure_start_global or sequence_start_global >= buffer_after_end_global):
                        in_excluded_zone = True
                        break

                elif TASK_MODE == 'detection':
                    # Detection mode: Exclude full 60-min buffer before and after seizure
                    # Buffer zone 1: [seizure_start - 60min, seizure_start) - before seizure
                    buffer_before_start_global = max(0, seizure_start_global - INTERICTAL_BUFFER)

                    # Check if sequence overlaps with pre-ictal buffer
                    if not (sequence_end_global <= buffer_before_start_global or sequence_start_global >= seizure_start_global):
                        in_excluded_zone = True
                        break

                    # Buffer zone 2: [seizure_start, seizure_end + 60min] - during and after seizure
                    buffer_after_end_global = seizure_end_global + INTERICTAL_BUFFER

                    # Check if sequence overlaps with ictal + post-ictal buffer
                    if not (sequence_end_global <= seizure_start_global or sequence_start_global >= buffer_after_end_global):
                        in_excluded_zone = True
                        break

        # Skip sequences in excluded buffer zone
        if in_excluded_zone:
            # Use non-overlapping stride for interictal regions (safest assumption)
            sequence_start_local += non_overlapping_stride
            continue

        # Create sequence metadata (store LOCAL times for file-based processing)
        sequence = {
            'patient_id': patient_id,
            'file': filename,
            'sequence_start_sec': sequence_start_local,
            'sequence_end_sec': sequence_end_local,
            'sequence_duration_sec': sequence_duration,
            'segment_starts': segment_starts,
            'num_segments': SEQUENCE_LENGTH,
            'type': sequence_type,
            'task_mode': TASK_MODE  # Track which mode generated this sequence
        }

        if sequence_type == 'preictal':
            sequence['time_to_seizure'] = time_to_seizure
        if matched_seizure_id is not None:
            sequence['seizure_id'] = matched_seizure_id
        else:
            sequence['seizure_id'] = None

        sequences.append(sequence)

        # Move to next sequence with adaptive stride
        # Use overlapping stride for sequences in positive regions (preictal/ictal)
        # Use non-overlapping stride for interictal sequences to prevent data leakage
        if is_in_positive_region(sequence_start_global, sequence_end_global):
            sequence_start_local += overlapping_stride  # 25s: 83% overlap for preictal/ictal
        else:
            sequence_start_local += non_overlapping_stride  # 150s: 0% overlap for interictal

    # Log skipped sequences if verbose warnings enabled
    if VERBOSE_WARNINGS and skipped_boundary_sequences > 0:
        print(f"  {filename}: Skipped {skipped_boundary_sequences} sequences due to insufficient boundary data (safety margin: {SAFETY_MARGIN}s)")

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
        seizures, all_files, file_times = parse_summary_file(summary_path)

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

        # Build cumulative timeline accounting for inter-file gaps
        file_timeline = {}  # Maps filename to {offset, duration, gap_before}

        def time_str_to_seconds(time_str):
            """Convert HH:MM:SS to seconds since midnight"""
            parts = time_str.split(':')
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

        # Build timeline for files with time information
        cumulative_offset = 0
        prev_end_time_sec = None

        for filename in all_files:
            if filename not in valid_files:
                continue

            edf_path = f"{BASE_PATH}{patient_id}/{filename}"
            if not os.path.exists(edf_path):
                continue

            # Get actual file duration
            file_duration = get_file_duration(edf_path)

            # Calculate gap from previous file (if both have time info)
            gap_before = 0
            if filename in file_times and 'start_time' in file_times[filename]:
                if prev_end_time_sec is not None and 'start_time' in file_times[filename]:
                    start_time_sec = time_str_to_seconds(file_times[filename]['start_time'])
                    gap_before = start_time_sec - prev_end_time_sec

                    # Handle day wraparound (if gap is negative, file is from next day)
                    if gap_before < 0:
                        gap_before += 24 * 3600  # Add 24 hours

                # Update prev_end_time_sec for next iteration
                if 'end_time' in file_times[filename]:
                    prev_end_time_sec = time_str_to_seconds(file_times[filename]['end_time'])

            # Store file info in timeline
            file_timeline[filename] = {
                'offset': cumulative_offset,
                'duration': file_duration,
                'gap_before': gap_before
            }

            # Update cumulative offset (add file duration + gap to next file)
            cumulative_offset += file_duration + gap_before

        # Convert all seizures to global timeline
        seizures_global = []
        for seizure in seizures:
            if seizure['file'] in file_timeline:
                file_offset = file_timeline[seizure['file']]['offset']
                seizure_id = len(seizures_global)
                seizures_global.append({
                    'file': seizure['file'],
                    'start_sec_local': seizure['start_sec'],
                    'end_sec_local': seizure['end_sec'],
                    'start_sec_global': file_offset + seizure['start_sec'],
                    'end_sec_global': file_offset + seizure['end_sec'],
                    'duration_sec': seizure['duration_sec'],
                    'seizure_id': seizure_id
                })

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

            # Skip files not in timeline (shouldn't happen but be safe)
            if filename not in file_timeline:
                continue

            file_info = file_timeline[filename]

            # Create sequences from this file (pass global seizures and file offset)
            sequences = create_sequences_from_file(
                patient_id,
                filename,
                seizures_global,  # Pass ALL seizures with global times
                file_info['duration'],
                file_info['offset']  # Pass this file's global offset
            )
            all_sequences.extend(sequences)

        # Count positive class (preictal or ictal) and interictal
        if TASK_MODE == 'prediction':
            positive_count = len([s for s in all_sequences if s['type'] == 'preictal'])
            positive_label = 'preictal'
        else:  # detection mode
            positive_count = len([s for s in all_sequences if s['type'] == 'ictal'])
            positive_label = 'ictal'

        interictal_count = len([s for s in all_sequences if s['type'] == 'interictal'])

        print(f"Patient {patient_id}: {len(all_sequences)} sequences ({positive_count} {positive_label}, {interictal_count} interictal)")

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

    # Process patients chb01 to chb24, skipping chb12 and chb24
    for i in range(1, 25):
        if i == 12:  # Skip patient 12 (insufficient data)
            continue
        if i == 24:  # Skip patient 24 (no interictal sequences due to missing file times)
            continue

        patient_id = f"chb{i:02d}"  # Format as chb01, chb02, etc.

        print(f"\nProcessing {patient_id}...")
        sequences, validation_results = create_sequences_single_patient(patient_id)
        all_sequences.extend(sequences)

        if validation_results:
            all_validation_results.append(validation_results)

    return all_sequences, all_validation_results

def balance_sequences_per_patient(sequences, random_seed=42):
    """Balance sequences per patient by downsampling interictal to match positive class

    For each patient, randomly removes interictal sequences until the count
    matches the positive class (preictal/ictal) count. This ensures 1:1 balance within each patient.

    Args:
        sequences: List of sequence dictionaries
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (balanced_sequences, balancing_stats)
    """
    random.seed(random_seed)

    # Determine positive class label based on mode
    positive_label = 'preictal' if TASK_MODE == 'prediction' else 'ictal'

    # Group sequences by patient
    patient_sequences = {}
    for seq in sequences:
        patient_id = seq['patient_id']
        if patient_id not in patient_sequences:
            patient_sequences[patient_id] = {positive_label: [], 'interictal': []}
        patient_sequences[patient_id][seq['type']].append(seq)

    balanced_sequences = []
    balancing_stats = {
        'patients': {},
        'total_original': len(sequences),
        'total_balanced': 0,
        'total_removed': 0,
        'patients_processed': 0,
        'positive_label': positive_label
    }

    # Balance each patient
    for patient_id in sorted(patient_sequences.keys()):
        positive = patient_sequences[patient_id].get(positive_label, [])
        interictal = patient_sequences[patient_id].get('interictal', [])

        n_positive = len(positive)
        n_interictal = len(interictal)

        # Keep all positive sequences
        balanced_sequences.extend(positive)

        # Randomly sample interictal to match positive count
        if n_interictal > n_positive:
            # Downsample interictal
            sampled_interictal = random.sample(interictal, n_positive)
            balanced_sequences.extend(sampled_interictal)
            removed = n_interictal - n_positive
        else:
            # Keep all interictal if fewer than positive
            balanced_sequences.extend(interictal)
            removed = 0

        # Track statistics
        balancing_stats['patients'][patient_id] = {
            f'{positive_label}_original': n_positive,
            'interictal_original': n_interictal,
            'interictal_kept': min(n_interictal, n_positive),
            'interictal_removed': removed,
            'total_balanced': n_positive + min(n_interictal, n_positive)
        }
        balancing_stats['total_removed'] += removed
        balancing_stats['patients_processed'] += 1

    balancing_stats['total_balanced'] = len(balanced_sequences)

    return balanced_sequences, balancing_stats

def assign_single_patient_splits(
    sequences,
    seizure_splits,
    interictal_ratios,
    fold_id,
    random_seed=None
):
    """Assign sequences for a single patient experiment into train/val/test splits.

    Args:
        sequences: List of sequence dictionaries for a single patient.
        seizure_splits: Dict mapping split names to lists of seizure_ids.
        interictal_ratios: Dict with ratios for distributing interictal sequences.
        fold_id: The fold ID for LOOCV (which seizure to hold out for testing)
        random_seed: Seed for deterministic interictal shuffling.

    Returns:
        Tuple of (retained_sequences, split_counts, dropped_positive_sequences, balance_stats)
    """
    # LOOCV mode: compute splits by seizure index
    required_splits = ['train', 'test']  # No validation in LOOCV mode
    # Compute LOOCV splits: test=fold_id, train=all others
    test_seizure = fold_id
    train_seizures = [i for i in range(LOOCV_TOTAL_SEIZURES) if i != test_seizure]
    seizure_splits = {
        'train': train_seizures,
        'test': [test_seizure]
    }
    print(f"LOOCV Fold {fold_id}: Test seizure={test_seizure}, Train seizures={train_seizures}")

    positive_label = 'preictal' if TASK_MODE == 'prediction' else 'ictal'
    splits = {split_name: [] for split_name in required_splits}
    interictal_pool = []
    dropped_positive = 0

    # Assign positive sequences based on seizure mapping
    for sequence in sequences:
        if sequence['type'] == positive_label:
            seizure_id = sequence.get('seizure_id')
            assigned_split = next(
                (split_name for split_name, seizure_ids in seizure_splits.items() if seizure_id in seizure_ids),
                None
            )
            if assigned_split is None:
                dropped_positive += 1
                continue

            sequence['split'] = assigned_split
            splits[assigned_split].append(sequence)
        else:
            interictal_pool.append(sequence)

    # Distribute interictal sequences according to requested ratios
    if interictal_pool:
        rng = random.Random(random_seed)
        rng.shuffle(interictal_pool)

        ratio_sum = sum(max(interictal_ratios.get(split_name, 0.0), 0.0) for split_name in required_splits)
        if ratio_sum <= 0:
            normalized = {split_name: 1.0 / len(required_splits) for split_name in required_splits}
        else:
            normalized = {
                split_name: max(interictal_ratios.get(split_name, 0.0), 0.0) / ratio_sum
                for split_name in required_splits
            }

        total_interictal = len(interictal_pool)
        counts = {}
        assigned_so_far = 0
        for idx, split_name in enumerate(required_splits):
            if idx == len(required_splits) - 1:
                count = total_interictal - assigned_so_far
            else:
                count = int(round(normalized[split_name] * total_interictal))
                count = max(0, min(count, total_interictal - assigned_so_far))
            counts[split_name] = count
            assigned_so_far += count

        start_idx = 0
        for split_name in required_splits:
            count = counts[split_name]
            end_idx = start_idx + count
            for sequence in interictal_pool[start_idx:end_idx]:
                sequence['split'] = split_name
                splits[split_name].append(sequence)
            start_idx = end_idx

        # Assign any leftover sequences (due to rounding) to the training split
        for sequence in interictal_pool[start_idx:]:
            sequence['split'] = required_splits[0]
            splits[required_splits[0]].append(sequence)

    # Balance each split by downsampling majority class (positive/interictal)
    balanced_splits, balance_stats = balance_sequences_across_splits(
        splits,
        positive_label,
        random_seed
    )

    retained_sequences = []
    for split_name in required_splits:
        retained_sequences.extend(balanced_splits[split_name])

    split_counts = {
        split_name: {
            'total': len(split_sequences),
            positive_label: sum(1 for seq in split_sequences if seq['type'] == positive_label),
            'interictal': sum(1 for seq in split_sequences if seq['type'] == 'interictal')
        }
        for split_name, split_sequences in balanced_splits.items()
    }

    return retained_sequences, split_counts, dropped_positive, balance_stats

def balance_sequences_across_splits(splits, positive_label, random_seed=42):
    """Balance each split by downsampling the majority class to match the minority.

    Args:
        splits: Dict mapping split name to list of sequences (each with 'type' label).
        positive_label: Positive class label ('preictal' or 'ictal').
        random_seed: Seed for deterministic shuffling.

    Returns:
        Tuple of (balanced_splits_dict, balance_stats)
    """
    rng = random.Random(random_seed)
    balanced_splits = {}
    balance_stats = {}

    for split_name, split_sequences in splits.items():
        positives = [seq for seq in split_sequences if seq['type'] == positive_label]
        interictals = [seq for seq in split_sequences if seq['type'] == 'interictal']
        others = [seq for seq in split_sequences if seq['type'] not in (positive_label, 'interictal')]

        if not positives or not interictals:
            balanced_splits[split_name] = split_sequences[:]
            balance_stats[split_name] = {
                'positive_kept': len(positives),
                'positive_dropped': 0,
                'interictal_kept': len(interictals),
                'interictal_dropped': 0,
                'other_sequences': len(others),
                'balanced': False,
                'note': 'Skipped balancing due to missing class'
            }
            continue

        target_count = min(len(positives), len(interictals))

        rng.shuffle(positives)
        rng.shuffle(interictals)

        kept_positive = positives[:target_count]
        kept_interictal = interictals[:target_count]
        dropped_positive = max(0, len(positives) - target_count)
        dropped_interictal = max(0, len(interictals) - target_count)

        combined = kept_positive + kept_interictal
        rng.shuffle(combined)

        if others:
            # Preserve other sequence types (if any) at the end for completeness
            combined.extend(others)

        balanced_splits[split_name] = combined
        balance_stats[split_name] = {
            'positive_kept': len(kept_positive),
            'positive_dropped': dropped_positive,
            'interictal_kept': len(kept_interictal),
            'interictal_dropped': dropped_interictal,
            'other_sequences': len(others),
            'balanced': True,
            'target_per_class': target_count
        }

    return balanced_splits, balance_stats

def save_sequences_to_file(
    sequences,
    validation_results,
    output_file=None,
    balancing_stats=None,
    split_summary=None,
    extra_summary=None
):
    """Save all sequences to a single file with validation information

    Args:
        sequences: List of sequence dictionaries
        validation_results: List of validation result dictionaries from each patient
        output_file: Output JSON filename (auto-generated based on mode if None)
        balancing_stats: Optional balancing statistics
        split_summary: Optional per-split counts (for single-patient mode)
        extra_summary: Optional dict of additional summary fields
    """

    # Determine positive class label and auto-generate filename if needed
    positive_label = 'preictal' if TASK_MODE == 'prediction' else 'ictal'
    if output_file is None:
        output_file = f"{OUTPUT_PREFIX}_sequences_{TASK_MODE}.json"

    # Separate by type
    positive_sequences = [s for s in sequences if s['type'] == positive_label]
    interictal_sequences = [s for s in sequences if s['type'] == 'interictal']

    # Get validation summary
    validation_summary = get_validation_summary(validation_results) if validation_results else {}

    data = {
        'sequences': sequences,
        f'{positive_label}_sequences': positive_sequences,
        'interictal_sequences': interictal_sequences,
        'summary': {
            'task_mode': TASK_MODE,
            'total_sequences': len(sequences),
            f'total_{positive_label}': len(positive_sequences),
            'total_interictal': len(interictal_sequences),
            'patients_processed': len(set([s['patient_id'] for s in sequences])),
            'segment_duration': SEGMENT_DURATION,
            'sequence_length': SEQUENCE_LENGTH,
            'sequence_stride': SEQUENCE_STRIDE,
            'sequence_total_duration': SEGMENT_DURATION * SEQUENCE_LENGTH,
            'preictal_window': PREICTAL_WINDOW if TASK_MODE == 'prediction' else 'N/A',
            'interictal_buffer': INTERICTAL_BUFFER,
            'class_balance': len(positive_sequences) / len(interictal_sequences) if interictal_sequences else 0
        },
        'validation_info': validation_summary
    }

    # Add balancing info if provided
    if balancing_stats:
        data['balancing_info'] = balancing_stats

    if split_summary:
        data['summary']['per_split'] = split_summary

    if extra_summary:
        data['summary'].update(extra_summary)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSequences saved to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

def print_summary(sequences):
    """Print comprehensive summary"""

    # Determine positive class label
    positive_label = 'preictal' if TASK_MODE == 'prediction' else 'ictal'
    positive_sequences = [s for s in sequences if s['type'] == positive_label]
    interictal_sequences = [s for s in sequences if s['type'] == 'interictal']

    # Overall summary
    print(f"\n=== OVERALL SUMMARY ===")
    print(f"Task mode: {TASK_MODE.upper()}")
    print(f"Total sequences: {len(sequences)}")
    print(f"  - {positive_label.capitalize()} sequences: {len(positive_sequences)}")
    print(f"  - Interictal sequences: {len(interictal_sequences)}")
    print(f"Sequence configuration:")
    print(f"  - Segments per sequence: {SEQUENCE_LENGTH}")
    print(f"  - Sequence duration: {SEGMENT_DURATION * SEQUENCE_LENGTH}s ({SEGMENT_DURATION * SEQUENCE_LENGTH / 60:.1f} min)")
    print(f"  - Sequence stride: {SEQUENCE_STRIDE} segments ({SEGMENT_DURATION * SEQUENCE_STRIDE}s)")
    if TASK_MODE == 'prediction':
        print(f"  - Preictal window: {PREICTAL_WINDOW//60} min")
    print(f"  - Interictal buffer: {INTERICTAL_BUFFER//60} min")

    # Per-patient breakdown
    patients = set([s['patient_id'] for s in sequences])
    print(f"\n=== PER-PATIENT BREAKDOWN ===")
    print(f"Patients processed: {len(patients)}")

    for patient in sorted(patients):
        patient_seqs = [s for s in sequences if s['patient_id'] == patient]
        positive_count = len([s for s in patient_seqs if s['type'] == positive_label])
        interictal_count = len([s for s in patient_seqs if s['type'] == 'interictal'])
        print(f"{patient}: {len(patient_seqs)} total ({positive_count} {positive_label}, {interictal_count} interictal)")

    # Per-split breakdown (only when split assignments exist)
    if any(seq.get('split') for seq in sequences):
        print(f"\n=== PER-SPLIT BREAKDOWN ===")
        available_splits = sorted({seq.get('split') for seq in sequences if seq.get('split')})
        for split_name in available_splits:
            split_sequences = [s for s in sequences if s.get('split') == split_name]
            split_positive = len([s for s in split_sequences if s['type'] == positive_label])
            split_interictal = len([s for s in split_sequences if s['type'] == 'interictal'])
            print(f"{split_name}: {len(split_sequences)} total ({split_positive} {positive_label}, {split_interictal} interictal)")

    # Statistics
    print(f"\n=== STATISTICS ===")
    if TASK_MODE == 'prediction' and positive_sequences:
        avg_time_to_seizure = sum([s.get('time_to_seizure', 0) for s in positive_sequences]) / len(positive_sequences)
        print(f"Average time to seizure (preictal): {avg_time_to_seizure:.1f}s ({avg_time_to_seizure/60:.1f} min)")

    print(f"Class balance: {len(positive_sequences)}/{len(interictal_sequences)} = {len(positive_sequences)/len(interictal_sequences) if interictal_sequences else 0:.2f}")

if __name__ == "__main__":
    try:
        # Determine which folds to process
        if LOOCV_FOLD_ID is None:
            folds_to_process = list(range(LOOCV_TOTAL_SEIZURES))
            print("="*60)
            print("BATCH PROCESSING: ALL FOLDS")
            print(f"Processing {len(folds_to_process)} folds for patient {SINGLE_PATIENT_ID}")
            print("="*60)
        else:
            folds_to_process = [LOOCV_FOLD_ID]
            print("="*60)
            print("SINGLE FOLD PROCESSING")
            print(f"Processing fold {LOOCV_FOLD_ID} for patient {SINGLE_PATIENT_ID}")
            print("="*60)

        print(f"\nConfiguration:")
        print(f"  - Task mode: {TASK_MODE.upper()} ({'preictal' if TASK_MODE == 'prediction' else 'ictal'} vs interictal)")
        print(f"  - LOOCV Mode: ENABLED")
        print(f"  - Sequence length: {SEQUENCE_LENGTH} segments")
        print(f"  - Sequence duration: {SEGMENT_DURATION * SEQUENCE_LENGTH}s ({SEGMENT_DURATION * SEQUENCE_LENGTH / 60:.1f} min)")
        overlap_pct = ((SEQUENCE_LENGTH - SEQUENCE_STRIDE) / SEQUENCE_LENGTH) * 100
        print(f"  - Stride: {SEQUENCE_STRIDE} segments ({overlap_pct:.0f}% overlap)")
        print(f"  - Segment duration: {SEGMENT_DURATION}s")
        if TASK_MODE == 'prediction':
            print(f"  - Preictal window: {PREICTAL_WINDOW // 60} min")
        print(f"  - Interictal buffer: {INTERICTAL_BUFFER // 60} min")
        print(f"  - Channel validation: {'ENABLED' if not SKIP_CHANNEL_VALIDATION else 'DISABLED'}")
        print(f"  - Target channels: {len(TARGET_CHANNELS)} channels")
        print("="*60)

        # Extract sequences once (shared across all folds)
        print(f"\nExtracting sequences from patient {SINGLE_PATIENT_ID}...")
        sequences, validation_results = create_sequences_single_patient(SINGLE_PATIENT_ID)
        validation_list = [validation_results] if validation_results else None

        if not sequences:
            print(f"❌ No sequences generated for patient {SINGLE_PATIENT_ID}!")
        else:
            # Process each fold
            for current_fold in folds_to_process:
                fold_config = get_fold_config(current_fold)
                current_output_prefix = fold_config['output_prefix']
                current_random_seed = fold_config['random_seed']

                print("\n" + "="*60)
                print(f"PROCESSING FOLD {current_fold}/{LOOCV_TOTAL_SEIZURES-1}")
                print("="*60)
                print(f"Output prefix: {current_output_prefix}")

                sequences_with_splits, split_counts, dropped_positive, balance_stats = assign_single_patient_splits(
                    sequences,
                    {},  # Not used in LOOCV mode (seizure splits computed automatically)
                    {},  # Not used in LOOCV mode (interictal ratios computed automatically)
                    fold_id=current_fold,
                    random_seed=current_random_seed
                )

                if not sequences_with_splits:
                    print("❌ No sequences retained after applying single-patient split configuration!")
                else:
                    positive_label = 'preictal' if TASK_MODE == 'prediction' else 'ictal'
                    print("\n" + "="*60)
                    print("FOLD SPLIT SUMMARY")
                    print("="*60)
                    print_summary(sequences_with_splits)

                    if dropped_positive:
                        print(f"\n⚠️ Dropped {dropped_positive} positive sequences because their seizure_id was not assigned to a split.")

                    if balance_stats:
                        print(f"\n=== SPLIT BALANCING SUMMARY ===")
                        split_names = ['train', 'test']  # LOOCV mode only (2-split configuration)
                        for split_name in split_names:
                            stats = balance_stats.get(split_name, {})
                            if not stats:
                                continue
                            if stats.get('balanced', False):
                                print(
                                    f"{split_name}: {stats['positive_kept']} {positive_label}, "
                                    f"{stats['interictal_kept']} interictal "
                                    f"(dropped {stats['positive_dropped']} {positive_label}, "
                                    f"{stats['interictal_dropped']} interictal)"
                                )
                            else:
                                print(f"{split_name}: {stats.get('note', 'Balancing skipped')}")

                    output_filename = f"{current_output_prefix}_sequences_{TASK_MODE}.json"
                    extra_summary = {'dropped_positive_sequences': dropped_positive} if dropped_positive else {}
                    if balance_stats:
                        extra_summary['split_balance'] = balance_stats
                    if not extra_summary:
                        extra_summary = None

                    save_sequences_to_file(
                        sequences_with_splits,
                        validation_list,
                        output_file=output_filename,
                        split_summary=split_counts,
                        extra_summary=extra_summary
                    )

                    if validation_list:
                        validation_summary = get_validation_summary(validation_list)
                        print(f"\n=== CHANNEL VALIDATION SUMMARY ===")
                        print(f"Validation enabled: {validation_summary['validation_enabled']}")
                        print(f"Files checked: {validation_summary['total_files_checked']}")
                        print(f"Files with valid channels: {validation_summary['files_with_valid_channels']}")
                        print(f"Files with invalid channels: {validation_summary['files_with_invalid_channels']}")

                        if validation_summary['files_with_invalid_channels'] > 0:
                            print(f"\nMost common missing channels:")
                            for channel, count in list(validation_summary['missing_channel_frequency'].items())[:5]:
                                print(f"  - {channel}: missing in {count} files")

                    print(f"\n✅ Fold {current_fold} completed successfully!")
                    print(f"✅ Sequences saved to {output_filename}")

            # Final summary
            if LOOCV_FOLD_ID is None:
                print("\n" + "="*60)
                print(f"✅ BATCH PROCESSING COMPLETED!")
                print(f"✅ Processed {len(folds_to_process)} folds for patient {SINGLE_PATIENT_ID}")
                print("="*60)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
