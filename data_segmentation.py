"""Sequence-based segmentation for CNN-LSTM model

Single Patient Training:
- Extracts sequences from all patients
- Each patient is processed independently

Channel validation happens here (not in preprocessing) to fail fast.
"""

import os
import json
import mne
import random
from data_segmentation_helpers.config import (
    TASK_MODE,
    SEGMENT_DURATION,
    SEQUENCE_LENGTH,
    SEQUENCE_STRIDE,
    PREICTAL_WINDOW,
    INTERICTAL_BUFFER,
    BASE_PATH,
    ESTIMATED_FILE_DURATION,
    VERBOSE_WARNINGS,
    TARGET_CHANNELS,
    SKIP_CHANNEL_VALIDATION,
    PATIENTS,
    PATIENT_INDEX,
    get_patient_config,
    SEIZURE_COUNTS,
    INTERICTAL_TO_PREICTAL_RATIO,
)
from data_segmentation_helpers.segmentation import parse_summary_file
from data_segmentation_helpers.channel_validation import (
    validate_patient_files,
    get_validation_summary,
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
        if TASK_MODE == "prediction":
            # Prediction mode: preictal window before seizure
            seizure_start_global = seizure["start_sec_global"]
            preictal_window_start_global = max(
                0, seizure_start_global - PREICTAL_WINDOW
            )
            # Region: from preictal window start to seizure start
            positive_regions.append(
                (preictal_window_start_global, seizure_start_global)
            )

        elif TASK_MODE == "detection":
            # Detection mode: during seizure
            seizure_start_global = seizure["start_sec_global"]
            seizure_end_global = seizure["end_sec_global"]
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


def create_sequences_from_file(
    patient_id, filename, all_seizures_global, file_duration, file_offset
):
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
    skipped_boundary_sequences = (
        0  # Track sequences skipped due to insufficient boundary data
    )

    # Calculate sequence parameters
    sequence_duration = SEGMENT_DURATION * SEQUENCE_LENGTH  # e.g., 5s * 30 = 150s
    overlapping_stride = (
        SEGMENT_DURATION * SEQUENCE_STRIDE
    )  # e.g., 5s * 5 = 25s (83% overlap for preictal/ictal)
    non_overlapping_stride = sequence_duration  # e.g., 150s (0% overlap for interictal)

    # Identify positive regions (preictal/ictal) to determine stride strategy
    positive_regions = identify_positive_regions(all_seizures_global)

    # Calculate how many sequences we can extract
    if file_duration < sequence_duration + (2 * SAFETY_MARGIN):
        return (
            sequences  # File too short for even one sequence (including safety margins)
        )

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
        segment_starts = [
            sequence_start_local + (i * SEGMENT_DURATION)
            for i in range(SEQUENCE_LENGTH)
        ]

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
        sequence_type = "interictal"
        time_to_seizure = None
        in_excluded_zone = False
        matched_seizure_id = None

        # PASS 1: Check for positive class (preictal OR ictal depending on mode)
        if TASK_MODE == "prediction":
            # Prediction mode: Check if preictal for ANY seizure (takes priority)
            for seizure in all_seizures_global:
                seizure_start_global = seizure["start_sec_global"]

                # Check if last segment falls in preictal window (10 minutes before seizure)
                preictal_window_start_global = max(
                    0, seizure_start_global - PREICTAL_WINDOW
                )

                if (
                    preictal_window_start_global <= last_segment_start_global
                    and last_segment_end_global <= seizure_start_global
                ):
                    sequence_type = "preictal"
                    time_to_seizure = seizure_start_global - last_segment_end_global
                    matched_seizure_id = seizure.get("seizure_id")
                    break

        elif TASK_MODE == "detection":
            # Detection mode: Check if ictal (overlaps with ANY seizure period)
            for seizure in all_seizures_global:
                seizure_start_global = seizure["start_sec_global"]
                seizure_end_global = seizure["end_sec_global"]

                # Check if ANY part of sequence overlaps with seizure period
                # Overlap occurs if: sequence_start < seizure_end AND sequence_end > seizure_start
                if (
                    sequence_start_global < seizure_end_global
                    and sequence_end_global > seizure_start_global
                ):
                    sequence_type = "ictal"
                    matched_seizure_id = seizure.get("seizure_id")
                    # Store overlap information (could be useful for analysis)
                    overlap_start = max(sequence_start_global, seizure_start_global)
                    overlap_end = min(sequence_end_global, seizure_end_global)
                    break

        # PASS 2: Only check buffer zones if NOT positive class (preictal/ictal)
        # Buffer zones should only exclude interictal sequences, not override positive labels
        if sequence_type == "interictal":
            for seizure in all_seizures_global:
                seizure_start_global = seizure["start_sec_global"]
                seizure_end_global = seizure["end_sec_global"]

                if TASK_MODE == "prediction":
                    # Prediction mode: Exclude 50-min buffer before preictal window
                    preictal_window_start_global = max(
                        0, seizure_start_global - PREICTAL_WINDOW
                    )

                    # Buffer zone 1: [seizure_start - 60min, seizure_start - 10min) - before preictal window
                    buffer_before_start_global = max(
                        0, seizure_start_global - INTERICTAL_BUFFER
                    )
                    buffer_before_end_global = preictal_window_start_global

                    # Check if sequence overlaps with pre-preictal buffer
                    if not (
                        sequence_end_global <= buffer_before_start_global
                        or sequence_start_global >= buffer_before_end_global
                    ):
                        in_excluded_zone = True
                        break

                    # Buffer zone 2: [seizure_start, seizure_end + 60min] - during and after seizure
                    buffer_after_end_global = seizure_end_global + INTERICTAL_BUFFER

                    # Check if sequence overlaps with ictal + post-ictal buffer
                    if not (
                        sequence_end_global <= seizure_start_global
                        or sequence_start_global >= buffer_after_end_global
                    ):
                        in_excluded_zone = True
                        break

                elif TASK_MODE == "detection":
                    # Detection mode: Exclude full 60-min buffer before and after seizure
                    # Buffer zone 1: [seizure_start - 60min, seizure_start) - before seizure
                    buffer_before_start_global = max(
                        0, seizure_start_global - INTERICTAL_BUFFER
                    )

                    # Check if sequence overlaps with pre-ictal buffer
                    if not (
                        sequence_end_global <= buffer_before_start_global
                        or sequence_start_global >= seizure_start_global
                    ):
                        in_excluded_zone = True
                        break

                    # Buffer zone 2: [seizure_start, seizure_end + 60min] - during and after seizure
                    buffer_after_end_global = seizure_end_global + INTERICTAL_BUFFER

                    # Check if sequence overlaps with ictal + post-ictal buffer
                    if not (
                        sequence_end_global <= seizure_start_global
                        or sequence_start_global >= buffer_after_end_global
                    ):
                        in_excluded_zone = True
                        break

        # Skip sequences in excluded buffer zone
        if in_excluded_zone:
            # Use non-overlapping stride for interictal regions (safest assumption)
            sequence_start_local += non_overlapping_stride
            continue

        # Create sequence metadata (store LOCAL times for file-based processing)
        sequence = {
            "patient_id": patient_id,
            "file": filename,
            "sequence_start_sec": sequence_start_local,
            "sequence_end_sec": sequence_end_local,
            "sequence_duration_sec": sequence_duration,
            "segment_starts": segment_starts,
            "num_segments": SEQUENCE_LENGTH,
            "type": sequence_type,
            "task_mode": TASK_MODE,  # Track which mode generated this sequence
        }

        if sequence_type == "preictal":
            sequence["time_to_seizure"] = time_to_seizure
        if matched_seizure_id is not None:
            sequence["seizure_id"] = matched_seizure_id
        else:
            sequence["seizure_id"] = None

        sequences.append(sequence)

        # Move to next sequence with adaptive stride
        # Use overlapping stride for sequences in positive regions (preictal/ictal)
        # Use non-overlapping stride for interictal sequences to prevent data leakage
        if is_in_positive_region(sequence_start_global, sequence_end_global):
            sequence_start_local += (
                overlapping_stride  # 25s: 83% overlap for preictal/ictal
            )
        else:
            sequence_start_local += (
                non_overlapping_stride  # 150s: 0% overlap for interictal
            )

    # Log skipped sequences if verbose warnings enabled
    if VERBOSE_WARNINGS and skipped_boundary_sequences > 0:
        print(
            f"  {filename}: Skipped {skipped_boundary_sequences} sequences due to insufficient boundary data (safety margin: {SAFETY_MARGIN}s)"
        )

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
        valid_files = set(validation_results["valid_file_list"])

        # Report validation results
        if validation_results["invalid_files"] > 0:
            print(
                f"  Channel validation: {validation_results['valid_files']}/{validation_results['total_files']} files valid"
            )
            if VERBOSE_WARNINGS:
                for invalid_info in validation_results["invalid_file_info"]:
                    print(
                        f"    Skipping {invalid_info['file']}: missing {invalid_info['missing_channels']}"
                    )

        # Build cumulative timeline accounting for inter-file gaps
        file_timeline = {}  # Maps filename to {offset, duration, gap_before}

        def time_str_to_seconds(time_str):
            """Convert HH:MM:SS to seconds since midnight"""
            parts = time_str.split(":")
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
            if filename in file_times and "start_time" in file_times[filename]:
                if (
                    prev_end_time_sec is not None
                    and "start_time" in file_times[filename]
                ):
                    start_time_sec = time_str_to_seconds(
                        file_times[filename]["start_time"]
                    )
                    gap_before = start_time_sec - prev_end_time_sec

                    # Handle day wraparound (if gap is negative, file is from next day)
                    if gap_before < 0:
                        gap_before += 24 * 3600  # Add 24 hours

                # Update prev_end_time_sec for next iteration
                if "end_time" in file_times[filename]:
                    prev_end_time_sec = time_str_to_seconds(
                        file_times[filename]["end_time"]
                    )

            # Store file info in timeline
            file_timeline[filename] = {
                "offset": cumulative_offset,
                "duration": file_duration,
                "gap_before": gap_before,
            }

            # Update cumulative offset (add file duration + gap to next file)
            cumulative_offset += file_duration + gap_before

        # Convert all seizures to global timeline
        seizures_global = []
        for seizure in seizures:
            if seizure["file"] in file_timeline:
                file_offset = file_timeline[seizure["file"]]["offset"]
                seizure_id = len(seizures_global)
                seizures_global.append(
                    {
                        "file": seizure["file"],
                        "start_sec_local": seizure["start_sec"],
                        "end_sec_local": seizure["end_sec"],
                        "start_sec_global": file_offset + seizure["start_sec"],
                        "end_sec_global": file_offset + seizure["end_sec"],
                        "duration_sec": seizure["duration_sec"],
                        "seizure_id": seizure_id,
                    }
                )

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
                file_info["duration"],
                file_info["offset"],  # Pass this file's global offset
            )
            all_sequences.extend(sequences)

        # Count positive class (preictal or ictal) and interictal
        if TASK_MODE == "prediction":
            positive_count = len([s for s in all_sequences if s["type"] == "preictal"])
            positive_label = "preictal"
        else:  # detection mode
            positive_count = len([s for s in all_sequences if s["type"] == "ictal"])
            positive_label = "ictal"

        interictal_count = len([s for s in all_sequences if s["type"] == "interictal"])

        print(
            f"Patient {patient_id}: {len(all_sequences)} sequences ({positive_count} {positive_label}, {interictal_count} interictal)"
        )

        return all_sequences, validation_results

    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        import traceback

        traceback.print_exc()
        return [], None


def assign_patient_splits(sequences, patient_id, random_seed=42):
    """Assign sequences to train/test splits for a single patient.

    Strategy:
    - Train set: All seizures except the last one (past)
    - Test set: The last seizure (future)

    Args:
        sequences: List of all sequence dictionaries for the patient
        patient_id: Patient ID
        random_seed: Seed for deterministic balancing

    Returns:
        Tuple of (retained_sequences, split_counts, balance_stats)
    """
    positive_label = "preictal" if TASK_MODE == "prediction" else "ictal"

    # 1. Get Patient data
    patient_seqs = [s for s in sequences if s["patient_id"] == patient_id]

    # 2. Identify unique seizures for this patient
    patient_seizure_ids = sorted(
        list(
            set(
                s["seizure_id"] for s in patient_seqs if s.get("seizure_id") is not None
            )
        )
    )
    num_seizures = len(patient_seizure_ids)

    # 3. Determine split: 70% Train, 30% Test (Chronological split)
    if num_seizures > 0:
        n_train_seizures = int(num_seizures * 0.7)
        # Ensure we don't have 0 training seizures if we have enough (e.g. 4+ seizures)
        # But for small counts (1, 2, 3), int(0.7) logic holds:
        # 1 -> 0 (0 train, 1 test)
        # 2 -> 1 (1 train, 1 test)
        # 3 -> 2 (2 train, 1 test)
    else:
        n_train_seizures = 0

    train_seizure_ids = set(patient_seizure_ids[:n_train_seizures])

    # 4. Find split point: where the first TEST seizure begins
    # Default to 0 if no training seizures (e.g. num_seizures is 0 or 1)
    split_index = 0
    if n_train_seizures > 0:
        split_index = len(patient_seqs)
        for i, seq in enumerate(patient_seqs):
            sid = seq.get("seizure_id")
            if sid is not None and sid not in train_seizure_ids:
                split_index = i
                break

    train_sequences = patient_seqs[:split_index]
    test_sequences = patient_seqs[split_index:]

    if num_seizures > 0:
        print(f"\nPatient {patient_id} Split:")
        print(f"  Total Seizures: {num_seizures}")
        print(
            f"  Train Seizures: {n_train_seizures} (IDs: {patient_seizure_ids[:n_train_seizures]})"
        )
        print(
            f"  Test Seizures: {num_seizures - n_train_seizures} (IDs: {patient_seizure_ids[n_train_seizures:]})"
        )
        print(f"  Sequences: {len(train_sequences)} Train / {len(test_sequences)} Test")

    # Check if test set has no interictal data and move from train if necessary
    test_interictal_count = sum(
        1 for seq in test_sequences if seq["type"] == "interictal"
    )

    if test_interictal_count == 0:
        print(
            f"  Warning: Test set has 0 interictal sequences. Moving sequences from train set..."
        )

        # Identify available interictal sequences in train
        train_interictal_indices = [
            i for i, seq in enumerate(train_sequences) if seq["type"] == "interictal"
        ]

        if train_interictal_indices:
            # Determine how many to move: match the number of positive sequences in test
            test_positive_count = sum(
                1 for seq in test_sequences if seq["type"] == positive_label
            )
            # Default to a small number if no positives (unlikely) or just 10
            num_to_move = (
                test_positive_count
                if test_positive_count > 0
                else min(10, len(train_interictal_indices))
            )

            # Don't take more than available
            num_to_move = min(num_to_move, len(train_interictal_indices))

            if num_to_move > 0:
                # Take from the end of the train set (closest to test set time-wise)
                indices_to_move = train_interictal_indices[-num_to_move:]
                indices_to_move_set = set(indices_to_move)

                # Extract sequences to move
                moved_sequences = [train_sequences[i] for i in indices_to_move]

                # Update lists: Remove from train, Add to test
                train_sequences = [
                    s
                    for i, s in enumerate(train_sequences)
                    if i not in indices_to_move_set
                ]
                test_sequences.extend(moved_sequences)

                print(
                    f"  Moved {len(moved_sequences)} interictal sequences from train to test."
                )
            else:
                print("  Warning: Not enough interictal data in train to move.")
        else:
            print("  Warning: Train set also has no interictal data!")

    # Assign split labels
    for seq in train_sequences:
        seq["split"] = "train"
    for seq in test_sequences:
        seq["split"] = "test"

    splits = {"train": train_sequences, "test": test_sequences}

    # Balance each split by downsampling majority class
    balanced_splits, balance_stats = balance_sequences_across_splits(
        splits, positive_label, random_seed
    )

    # Flatten sequences
    retained_sequences = []
    for split_name in ["train", "test"]:
        retained_sequences.extend(balanced_splits[split_name])

    # Compute split counts
    split_counts = {
        split_name: {
            "total": len(split_seqs),
            positive_label: sum(
                1 for seq in split_seqs if seq["type"] == positive_label
            ),
            "interictal": sum(1 for seq in split_seqs if seq["type"] == "interictal"),
        }
        for split_name, split_seqs in balanced_splits.items()
    }

    return retained_sequences, split_counts, balance_stats


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
        positives = [seq for seq in split_sequences if seq["type"] == positive_label]
        interictals = [seq for seq in split_sequences if seq["type"] == "interictal"]
        others = [
            seq
            for seq in split_sequences
            if seq["type"] not in (positive_label, "interictal")
        ]

        if not positives or not interictals:
            balanced_splits[split_name] = split_sequences[:]
            balance_stats[split_name] = {
                "positive_kept": len(positives),
                "positive_dropped": 0,
                "interictal_kept": len(interictals),
                "interictal_dropped": 0,
                "other_sequences": len(others),
                "balanced": False,
                "note": "Skipped balancing due to missing class",
            }
            continue

        # Determine target counts based on ratio
        # We want to keep all positives if possible
        target_positive = len(positives)
        
        # Calculate target interictals based on ratio
        target_interictal = int(len(positives) * INTERICTAL_TO_PREICTAL_RATIO)
        
        # Cap at available interictals
        target_interictal = min(target_interictal, len(interictals))

        rng.shuffle(positives)
        rng.shuffle(interictals)

        kept_positive = positives[:target_positive]
        kept_interictal = interictals[:target_interictal]
        dropped_positive = max(0, len(positives) - target_positive) # Should be 0
        dropped_interictal = max(0, len(interictals) - target_interictal)

        combined = kept_positive + kept_interictal
        rng.shuffle(combined)

        if others:
            # Preserve other sequence types (if any) at the end for completeness
            combined.extend(others)

        balanced_splits[split_name] = combined
        balance_stats[split_name] = {
            "positive_kept": len(kept_positive),
            "positive_dropped": dropped_positive,
            "interictal_kept": len(kept_interictal),
            "interictal_dropped": dropped_interictal,
            "other_sequences": len(others),
            "balanced": True,
            "target_per_class": target_interictal, # recording interictal target as ref
            "ratio": INTERICTAL_TO_PREICTAL_RATIO
        }

    return balanced_splits, balance_stats


def save_sequences_to_file(
    sequences,
    validation_results,
    output_file=None,
    balancing_stats=None,
    split_summary=None,
    extra_summary=None,
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

    # Determine positive class label
    positive_label = "preictal" if TASK_MODE == "prediction" else "ictal"
    if output_file is None:
        raise ValueError("output_file must be specified")

    # Separate by type
    positive_sequences = [s for s in sequences if s["type"] == positive_label]
    interictal_sequences = [s for s in sequences if s["type"] == "interictal"]

    # Get validation summary
    validation_summary = (
        get_validation_summary(validation_results) if validation_results else {}
    )

    data = {
        "sequences": sequences,
        f"{positive_label}_sequences": positive_sequences,
        "interictal_sequences": interictal_sequences,
        "summary": {
            "task_mode": TASK_MODE,
            "total_sequences": len(sequences),
            f"total_{positive_label}": len(positive_sequences),
            "total_interictal": len(interictal_sequences),
            "patients_processed": len(set([s["patient_id"] for s in sequences])),
            "segment_duration": SEGMENT_DURATION,
            "sequence_length": SEQUENCE_LENGTH,
            "sequence_stride": SEQUENCE_STRIDE,
            "sequence_total_duration": SEGMENT_DURATION * SEQUENCE_LENGTH,
            "preictal_window": PREICTAL_WINDOW if TASK_MODE == "prediction" else "N/A",
            "interictal_buffer": INTERICTAL_BUFFER,
            "class_balance": (
                len(positive_sequences) / len(interictal_sequences)
                if interictal_sequences
                else 0
            ),
        },
        "validation_info": validation_summary,
    }

    # Add balancing info if provided
    if balancing_stats:
        data["balancing_info"] = balancing_stats

    if split_summary:
        data["summary"]["per_split"] = split_summary

    if extra_summary:
        data["summary"].update(extra_summary)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSequences saved to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")


def print_summary(sequences):
    """Print comprehensive summary"""

    # Determine positive class label
    positive_label = "preictal" if TASK_MODE == "prediction" else "ictal"
    positive_sequences = [s for s in sequences if s["type"] == positive_label]
    interictal_sequences = [s for s in sequences if s["type"] == "interictal"]

    # Overall summary
    print(f"\n=== OVERALL SUMMARY ===")
    print(f"Task mode: {TASK_MODE.upper()}")
    print(f"Total sequences: {len(sequences)}")
    print(f"  - {positive_label.capitalize()} sequences: {len(positive_sequences)}")
    print(f"  - Interictal sequences: {len(interictal_sequences)}")
    print(f"Sequence configuration:")
    print(f"  - Segments per sequence: {SEQUENCE_LENGTH}")
    print(
        f"  - Sequence duration: {SEGMENT_DURATION * SEQUENCE_LENGTH}s ({SEGMENT_DURATION * SEQUENCE_LENGTH / 60:.1f} min)"
    )
    print(
        f"  - Sequence stride: {SEQUENCE_STRIDE} segments ({SEGMENT_DURATION * SEQUENCE_STRIDE}s)"
    )
    if TASK_MODE == "prediction":
        print(f"  - Preictal window: {PREICTAL_WINDOW//60} min")
    print(f"  - Interictal buffer: {INTERICTAL_BUFFER//60} min")

    # Per-patient breakdown
    patients = set([s["patient_id"] for s in sequences])
    print(f"\n=== PER-PATIENT BREAKDOWN ===")
    print(f"Patients processed: {len(patients)}")

    for patient in sorted(patients):
        patient_seqs = [s for s in sequences if s["patient_id"] == patient]
        positive_count = len([s for s in patient_seqs if s["type"] == positive_label])
        interictal_count = len([s for s in patient_seqs if s["type"] == "interictal"])
        print(
            f"{patient}: {len(patient_seqs)} total ({positive_count} {positive_label}, {interictal_count} interictal)"
        )

    # Per-split breakdown (only when split assignments exist)
    if any(seq.get("split") for seq in sequences):
        print(f"\n=== PER-SPLIT BREAKDOWN ===")
        available_splits = sorted(
            {seq.get("split") for seq in sequences if seq.get("split")}
        )
        for split_name in available_splits:
            split_sequences = [s for s in sequences if s.get("split") == split_name]
            split_positive = len(
                [s for s in split_sequences if s["type"] == positive_label]
            )
            split_interictal = len(
                [s for s in split_sequences if s["type"] == "interictal"]
            )
            print(
                f"{split_name}: {len(split_sequences)} total ({split_positive} {positive_label}, {split_interictal} interictal)"
            )

    # Statistics
    print(f"\n=== STATISTICS ===")
    if TASK_MODE == "prediction" and positive_sequences:
        avg_time_to_seizure = sum(
            [s.get("time_to_seizure", 0) for s in positive_sequences]
        ) / len(positive_sequences)
        print(
            f"Average time to seizure (preictal): {avg_time_to_seizure:.1f}s ({avg_time_to_seizure/60:.1f} min)"
        )

    print(
        f"Class balance: {len(positive_sequences)}/{len(interictal_sequences)} = {len(positive_sequences)/len(interictal_sequences) if interictal_sequences else 0:.2f}"
    )


if __name__ == "__main__":
    try:
        # =================================================================
        # SINGLE PATIENT PROCESSING
        # =================================================================

        # Validate configuration
        if not PATIENTS:
            raise ValueError("PATIENTS list is empty. Add patient IDs to config.py")

        print("=" * 60)
        print("SINGLE PATIENT PROCESSING")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(
            f"  - Task mode: {TASK_MODE.upper()} ({'preictal' if TASK_MODE == 'prediction' else 'ictal'} vs interictal)"
        )
        print(f"  - Total patients: {len(PATIENTS)}")
        print(f"  - Patients: {PATIENTS}")
        print(
            f"  - Sequence length: {SEQUENCE_LENGTH} segments ({SEGMENT_DURATION * SEQUENCE_LENGTH}s)"
        )
        overlap_pct = ((SEQUENCE_LENGTH - SEQUENCE_STRIDE) / SEQUENCE_LENGTH) * 100
        print(f"  - Stride: {SEQUENCE_STRIDE} segments ({overlap_pct:.0f}% overlap)")
        print(f"  - Segment duration: {SEGMENT_DURATION}s")
        if TASK_MODE == "prediction":
            print(f"  - Preictal window: {PREICTAL_WINDOW // 60} min")
        print(f"  - Interictal buffer: {INTERICTAL_BUFFER // 60} min")
        print(
            f"  - Channel validation: {'ENABLED' if not SKIP_CHANNEL_VALIDATION else 'DISABLED'}"
        )
        print(f"  - Target channels: {len(TARGET_CHANNELS)} channels")

        # Determine which patients to process
        n_patients = len(PATIENTS)
        if PATIENT_INDEX is None:
            patients_to_process = list(range(n_patients))
            print(f"\nProcessing ALL {len(patients_to_process)} patients")
        else:
            patients_to_process = [PATIENT_INDEX]
            patient_cfg = get_patient_config(PATIENT_INDEX)
            print(
                f"\nProcessing SINGLE patient {PATIENT_INDEX}: {patient_cfg['patient_id']}"
            )
        print("=" * 60)

        # =================================================================
        # PROCESS EACH PATIENT
        # =================================================================
        total_sequences = 0

        for current_idx in patients_to_process:
            patient_config = get_patient_config(current_idx)
            patient_id = patient_config["patient_id"]
            current_output_prefix = patient_config["output_prefix"]
            current_random_seed = patient_config["random_seed"]

            print("\n" + "=" * 60)
            print(f"PROCESSING PATIENT {current_idx}/{n_patients-1}: {patient_id}")
            print("=" * 60)
            print(f"  Output prefix: {current_output_prefix}")

            # 1. Create sequences for this patient
            sequences, validation_results = create_sequences_single_patient(patient_id)

            if not sequences:
                print(f"[WARNING] No sequences generated for {patient_id}")
                continue

            # 2. Assign sequences to train/test splits
            sequences_with_splits, split_counts, balance_stats = assign_patient_splits(
                sequences, patient_id, random_seed=current_random_seed
            )

            if not sequences_with_splits:
                print("[ERROR] No sequences retained after split assignment!")
                continue

            positive_label = "preictal" if TASK_MODE == "prediction" else "ictal"

            # Print split summary
            print(f"\n=== SPLIT SUMMARY ===")
            for split_name in ["train", "test"]:
                counts = split_counts.get(split_name, {})
                print(
                    f"  {split_name}: {counts.get('total', 0)} sequences "
                    f"({counts.get(positive_label, 0)} {positive_label}, "
                    f"{counts.get('interictal', 0)} interictal)"
                )

            # Print balance stats
            if balance_stats:
                print(f"\n=== BALANCING SUMMARY ===")
                for split_name in ["train", "test"]:
                    stats = balance_stats.get(split_name, {})
                    if stats.get("balanced", False):
                        print(
                            f"  {split_name}: kept {stats['positive_kept']} {positive_label}, "
                            f"{stats['interictal_kept']} interictal "
                            f"(dropped {stats['positive_dropped']} {positive_label}, "
                            f"{stats['interictal_dropped']} interictal)"
                        )
                    else:
                        print(
                            f"  {split_name}: {stats.get('note', 'Balancing skipped')}"
                        )

            # Save sequences to file
            output_filename = f"{current_output_prefix}_sequences_{TASK_MODE}.json"
            extra_summary = {
                "patient_index": current_idx,
                "patient_id": patient_id,
                "split_balance": balance_stats,
            }

            # Convert validation results to list format for save_sequences_to_file
            # (Function expects a list of dicts if multiple patients, or handling single dict)
            validation_list = [validation_results] if validation_results else None

            save_sequences_to_file(
                sequences_with_splits,
                validation_list,
                output_file=output_filename,
                split_summary=split_counts,
                extra_summary=extra_summary,
            )

            print(f"\n[SUCCESS] Patient {patient_id} completed: {output_filename}")
            total_sequences += len(sequences_with_splits)

        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        print("\n" + "=" * 60)
        print(f"[SUCCESS] PROCESSING COMPLETE")
        print(f"   Processed {len(patients_to_process)} patient(s)")
        print(f"   Total sequences generated: {total_sequences}")
        print("=" * 60)

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback

        traceback.print_exc()
