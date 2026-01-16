"""Sequence-based segmentation for CNN-LSTM model

Leave-One-Patient-Out (LOPO) cross-validation:
- Extracts sequences from all patients
- One patient held out for testing, all others for training

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
    LOPO_PATIENTS,
    LOPO_FOLD_ID,
    get_fold_config,
    SEIZURE_COUNTS,
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
                    preictal_window_start_global
                    <= last_segment_start_global
                    < seizure_start_global
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


def create_sequences_multi_patient(patient_ids):
    """Extract sequences from multiple patients for LOPO cross-validation.

    Args:
        patient_ids: List of patient IDs to process (e.g., ['chb01', 'chb02', ...])

    Returns:
        Tuple of (all_sequences, validation_results_by_patient)
            - all_sequences: List of all sequence dictionaries from all patients
            - validation_results_by_patient: Dict mapping patient_id to validation results
    """
    all_sequences = []
    validation_results_by_patient = {}

    print(f"\n{'='*60}")
    print(f"Extracting sequences from {len(patient_ids)} patients")
    print(f"{'='*60}")

    for patient_id in patient_ids:
        print(f"\nProcessing {patient_id}...")
        sequences, patient_validation = create_sequences_single_patient(patient_id)

        if sequences:
            all_sequences.extend(sequences)
            if patient_validation:
                validation_results_by_patient[patient_id] = patient_validation

            # Count by type
            positive_label = "preictal" if TASK_MODE == "prediction" else "ictal"
            pos_count = sum(1 for s in sequences if s["type"] == positive_label)
            int_count = len(sequences) - pos_count
            print(
                f"  -> {len(sequences)} sequences ({pos_count} {positive_label}, {int_count} interictal)"
            )
        else:
            print(f"  -> No sequences generated (check for missing files or channels)")

    # Summary
    print(f"\n{'='*60}")
    print(f"Total: {len(all_sequences)} sequences from {len(patient_ids)} patients")
    print(f"{'='*60}")

    return all_sequences, validation_results_by_patient


def assign_multi_patient_splits(sequences, test_patient_id, random_seed=42):
    """Assign sequences to train/test splits based on patient ID (LOPO).

    In Leave-One-Patient-Out cross-validation:
    - Train set: All sequences from all other patients + First ~50% of test patient's seizures
    - Test set: Remaining ~50% of test patient's seizures (future prediction)

    Args:
        sequences: List of all sequence dictionaries from all patients
        test_patient_id: Patient ID to hold out for testing
        random_seed: Seed for deterministic balancing

    Returns:
        Tuple of (retained_sequences, split_counts, balance_stats)
    """
    positive_label = "preictal" if TASK_MODE == "prediction" else "ictal"

    # 1. Separate other patients (always Train)
    other_train_sequences = [s for s in sequences if s["patient_id"] != test_patient_id]

    # 2. Get Test Patient data (chronologically ordered)
    patient_seqs = [s for s in sequences if s["patient_id"] == test_patient_id]

    # 3. Identify unique seizures for this patient
    patient_seizure_ids = sorted(
        list(set(s["seizure_id"] for s in patient_seqs if s.get("seizure_id") is not None))
    )
    num_seizures = len(patient_seizure_ids)

    # 4. Determine split: Train on ALL seizures except the LAST one
    # If there is only 1 seizure, n_train_seizures = 0 (standard LOPO for that seizure)
    if num_seizures > 0:
        n_train_seizures = num_seizures - 1
    else:
        n_train_seizures = 0
        
    train_seizure_ids = set(patient_seizure_ids[:n_train_seizures])

    # 5. Find split point: where the first TEST seizure begins
    # Default to 0 if no training seizures (e.g. num_seizures is 0 or 1)
    split_index = 0
    if n_train_seizures > 0:
        split_index = len(patient_seqs)
        for i, seq in enumerate(patient_seqs):
            sid = seq.get("seizure_id")
            if sid is not None and sid not in train_seizure_ids:
                split_index = i
                break

    patient_train = patient_seqs[:split_index]
    patient_test = patient_seqs[split_index:]

    if num_seizures > 0:
        print(f"\nTest Patient {test_patient_id} Split (Semi-LOPO):")
        print(f"  Total Seizures: {num_seizures}")
        print(
            f"  Train Seizures: {n_train_seizures} (IDs: {patient_seizure_ids[:n_train_seizures]})"
        )
        print(
            f"  Test Seizures: {num_seizures - n_train_seizures} (IDs: {patient_seizure_ids[n_train_seizures:]})"
        )
        print(f"  Sequences: {len(patient_train)} Train / {len(patient_test)} Test")

    # 6. Combine
    train_sequences = other_train_sequences + patient_train
    test_sequences = patient_test

    # Get unique patients in each split
    train_patients = sorted(set(s["patient_id"] for s in train_sequences))

    print(f"\nFinal Split Configuration:")
    print(f"  Train sequences: {len(train_sequences)} (from {len(train_patients)} patients)")
    print(f"  Test sequences: {len(test_sequences)} (from {test_patient_id})")

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
            positive_label: sum(1 for seq in split_seqs if seq["type"] == positive_label),
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
            "positive_kept": len(kept_positive),
            "positive_dropped": dropped_positive,
            "interictal_kept": len(kept_interictal),
            "interictal_dropped": dropped_interictal,
            "other_sequences": len(others),
            "balanced": True,
            "target_per_class": target_count,
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
        # LEAVE-ONE-PATIENT-OUT (LOPO) CROSS-VALIDATION
        # =================================================================

        # Validate configuration
        if not LOPO_PATIENTS:
            raise ValueError(
                "LOPO_PATIENTS list is empty. Add patient IDs to config.py"
            )

        print("=" * 60)
        print("LEAVE-ONE-PATIENT-OUT (LOPO) CROSS-VALIDATION")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(
            f"  - Task mode: {TASK_MODE.upper()} ({'preictal' if TASK_MODE == 'prediction' else 'ictal'} vs interictal)"
        )
        print(f"  - Total patients: {len(LOPO_PATIENTS)}")
        print(f"  - Patients: {LOPO_PATIENTS}")
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

        # Determine which folds to process
        n_folds = len(LOPO_PATIENTS)
        if LOPO_FOLD_ID is None:
            folds_to_process = list(range(n_folds))
            print(f"\nProcessing ALL {len(folds_to_process)} folds")
        else:
            folds_to_process = [LOPO_FOLD_ID]
            fold_cfg = get_fold_config(LOPO_FOLD_ID)
            print(
                f"\nProcessing SINGLE fold {LOPO_FOLD_ID}: test patient = {fold_cfg['test_patient']}"
            )
        print("=" * 60)

        # =================================================================
        # UPFRONT VALIDATION
        # =================================================================
        if True:  # Always validate upfront
            print(
                f"\nValidating all {len(LOPO_PATIENTS)} patients have required channels..."
            )
            validation_summary_upfront = {}
            patients_with_issues = []

            for patient_id in LOPO_PATIENTS:
                summary_path = f"{BASE_PATH}{patient_id}/{patient_id}-summary.txt"
                if not os.path.exists(summary_path):
                    print(
                        f"  WARNING: {patient_id} - Summary file not found: {summary_path}"
                    )
                    patients_with_issues.append(patient_id)
                    continue

                try:
                    seizures, all_files, _ = parse_summary_file(summary_path)
                    patient_validation = validate_patient_files(
                        patient_id, all_files, BASE_PATH
                    )
                    validation_summary_upfront[patient_id] = patient_validation

                    valid_ratio = (
                        patient_validation["valid_files"]
                        / max(patient_validation["total_files"], 1)
                        * 100
                    )
                    seizure_count = SEIZURE_COUNTS.get(patient_id, len(seizures))
                    print(
                        f"  {patient_id}: {patient_validation['valid_files']}/{patient_validation['total_files']} files valid ({valid_ratio:.0f}%), {seizure_count} seizures"
                    )

                    if patient_validation["valid_files"] == 0:
                        patients_with_issues.append(patient_id)
                except Exception as e:
                    print(f"  ERROR: {patient_id} - {e}")
                    patients_with_issues.append(patient_id)

            if patients_with_issues:
                print(
                    f"\n⚠️ Warning: {len(patients_with_issues)} patients have issues: {patients_with_issues}"
                )
                print(
                    "  Consider removing these patients from LOPO_PATIENTS in config.py"
                )
            print()

        # =================================================================
        # EXTRACT SEQUENCES FROM ALL PATIENTS (shared across all folds)
        # =================================================================
        sequences, validation_results_by_patient = create_sequences_multi_patient(
            LOPO_PATIENTS
        )

        if not sequences:
            print("❌ No sequences generated from any patient!")
            exit(1)

        # Convert validation results to list format for save_sequences_to_file
        validation_list = (
            list(validation_results_by_patient.values())
            if validation_results_by_patient
            else None
        )

        print(
            f"\n✓ Total: {len(sequences)} sequences extracted from {len(LOPO_PATIENTS)} patients"
        )

        # =================================================================
        # PROCESS EACH FOLD
        # =================================================================
        for current_fold in folds_to_process:
            fold_config = get_fold_config(current_fold)
            test_patient = fold_config["test_patient"]
            train_patients = fold_config["train_patients"]
            current_output_prefix = fold_config["output_prefix"]
            current_random_seed = fold_config["random_seed"]

            print("\n" + "=" * 60)
            print(f"FOLD {current_fold}/{n_folds-1}: Test patient = {test_patient}")
            print("=" * 60)
            print(f"  Train patients ({len(train_patients)}): {train_patients}")
            print(f"  Output prefix: {current_output_prefix}")

            # Assign sequences to train/test based on patient ID
            sequences_with_splits, split_counts, balance_stats = (
                assign_multi_patient_splits(
                    sequences, test_patient, random_seed=current_random_seed
                )
            )

            if not sequences_with_splits:
                print("❌ No sequences retained after split assignment!")
                continue

            positive_label = "preictal" if TASK_MODE == "prediction" else "ictal"

            # Print fold summary
            print(f"\n=== FOLD SPLIT SUMMARY ===")
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
                "fold_id": current_fold,
                "test_patient": test_patient,
                "train_patients": train_patients,
                "split_balance": balance_stats,
            }

            save_sequences_to_file(
                sequences_with_splits,
                validation_list,
                output_file=output_filename,
                split_summary=split_counts,
                extra_summary=extra_summary,
            )

            print(f"\n✅ Fold {current_fold} completed: {output_filename}")

        # =================================================================
        # FINAL SUMMARY
        # =================================================================
        print("\n" + "=" * 60)
        print(f"✅ LOPO CROSS-VALIDATION COMPLETE")
        print(f"   Processed {len(folds_to_process)} fold(s)")
        print(f"   Total patients: {len(LOPO_PATIENTS)}")
        print(f"   Total sequences: {len(sequences)}")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
