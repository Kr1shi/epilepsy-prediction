"""Sequence-based segmentation for CNN-LSTM model
Extracts sequences of consecutive 30-second segments for temporal modeling.

This module creates sequence metadata AND validates that files have all required
channels. Channel validation happens here (not in preprocessing).
"""

import os
import json
import mne
import random
from data_segmentation_helpers.config import *
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

    # Determine positive class label and auto-generate filename if needed
    positive_label = "preictal" if TASK_MODE == "prediction" else "ictal"
    if output_file is None:
        output_file = f"{OUTPUT_PREFIX}_sequences_{TASK_MODE}.json"

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


def create_sequences_lopo(test_patient_id, random_seed=42):
    """Create sequences for LOPO cross-validation
    
    Args:
        test_patient_id: Patient ID to use as test set
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (sequences, validation_results)
    """
    # Import here to avoid circular imports
    import argparse
    from data_segmentation_helpers.seizure_counts import SEIZURE_COUNTS
    
    sequences = []
    all_validation_results = []
    
    # Get all patient IDs
    all_patients = sorted(SEIZURE_COUNTS.keys())
    
    if test_patient_id not in all_patients:
        raise ValueError(f"Test patient {test_patient_id} not found in dataset")
    
    # Get training patients (all except test patient)
    train_patients = [p for p in all_patients if p != test_patient_id]
    
    print(f"\nLOPO Configuration:")
    print(f"  - Test patient: {test_patient_id}")
    print(f"  - Training patients: {len(train_patients)} patients")
    print(f"  - Random seed: {random_seed}")
    
    # Set random seed for reproducible test/train assignment
    rng = random.Random(random_seed)
    
    # Process all patients
    for patient_id in all_patients:
        print(f"\nProcessing patient {patient_id}...")
        
        # Parse summary file
        summary_path = f"{BASE_PATH}{patient_id}/{patient_id}-summary.txt"
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
        
        try:
            seizures, all_files, file_times = parse_summary_file(summary_path)
            
            # Validate channels for all files
            validation_results = validate_patient_files(patient_id, all_files, BASE_PATH)
            all_validation_results.append(validation_results)
            
            # Only process files with valid channels
            valid_files = set(validation_results["valid_file_list"])
            
            # Report validation results
            if validation_results["invalid_files"] > 0:
                print(
                    f"  Channel validation: {validation_results['valid_files']}/{validation_results['total_files']} files valid"
                )
            
            # Build file timeline
            file_timeline = {}
            
            def time_str_to_seconds(time_str):
                """Convert HH:MM:SS to seconds since midnight"""
                parts = time_str.split(":")
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            
            cumulative_offset = 0
            prev_end_time_sec = None
            
            for filename in all_files:
                if filename not in valid_files:
                    continue
                
                edf_path = f"{BASE_PATH}{patient_id}/{filename}"
                if not os.path.exists(edf_path):
                    continue
                
                file_duration = get_file_duration(edf_path)
                
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
                        
                        if gap_before < 0:
                            gap_before += 24 * 3600
                    
                    if "end_time" in file_times[filename]:
                        prev_end_time_sec = time_str_to_seconds(
                            file_times[filename]["end_time"]
                        )
                
                file_timeline[filename] = {
                    "offset": cumulative_offset,
                    "duration": file_duration,
                    "gap_before": gap_before,
                }
                
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
                            "seizure_id": seizure_id,
                        }
                    )
            
            # Create sequences for each file
            positive_regions = identify_positive_regions(seizures_global)
            
            for filename in all_files:
                if filename not in file_timeline:
                    continue
                
                file_offset = file_timeline[filename]["offset"]
                file_duration = file_timeline[filename]["duration"]
                
                file_sequences = create_sequences_from_file(
                    patient_id,
                    filename,
                    seizures_global,
                    file_duration,
                    file_offset,
                )
                sequences.extend(file_sequences)
        
        except Exception as e:
            print(f"  Error processing patient {patient_id}: {e}")
            raise
    
    # Assign train/test splits based on patient ID
    for sequence in sequences:
        if sequence['patient_id'] == test_patient_id:
            sequence['split'] = 'test'
        else:
            sequence['split'] = 'train'
    
    return sequences, all_validation_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract sequences for LOPO cross-validation")
    parser.add_argument('--test_patient', type=str, default=None,
                        help='Patient ID to test on (None = process all patients)')
    args = parser.parse_args()
    
    try:
        if args.test_patient is None:
            # Process all patients
            from data_segmentation_helpers.seizure_counts import SEIZURE_COUNTS
            all_patients = sorted(SEIZURE_COUNTS.keys())
            
            print("=" * 60)
            print("BATCH PROCESSING: ALL PATIENTS (LOPO)")
            print(f"Processing {len(all_patients)} patients")
            print("=" * 60)
            
            for test_patient_id in all_patients:
                print("\n" + "=" * 60)
                print(f"PROCESSING PATIENT {test_patient_id}")
                print("=" * 60)
                
                try:
                    sequences, validation_results = create_sequences_lopo(test_patient_id)
                    
                    if not sequences:
                        print(f"❌ No sequences generated for test patient {test_patient_id}!")
                        continue
                    
                    # Count splits
                    train_count = sum(1 for s in sequences if s['split'] == 'train')
                    test_count = sum(1 for s in sequences if s['split'] == 'test')
                    
                    print(f"\nSequence Summary:")
                    print(f"  - Training sequences: {train_count}")
                    print(f"  - Test sequences: {test_count}")
                    print(f"  - Total sequences: {len(sequences)}")
                    
                    # Save to file
                    output_filename = f"lopo_test_{test_patient_id}_sequences_{TASK_MODE}.json"
                    split_counts = {
                        'train': train_count,
                        'test': test_count,
                        'total': len(sequences)
                    }
                    
                    save_sequences_to_file(
                        sequences,
                        validation_results,
                        output_file=output_filename,
                        split_summary=split_counts,
                    )
                    
                    print(f"✅ Saved to {output_filename}")
                
                except Exception as e:
                    print(f"❌ Error processing patient {test_patient_id}: {e}")
                    import traceback
                    traceback.print_exc()
        
        else:
            # Process single test patient
            test_patient_id = args.test_patient
            
            print("=" * 60)
            print("LOPO PROCESSING: SINGLE PATIENT")
            print(f"Test patient: {test_patient_id}")
            print("=" * 60)
            
            try:
                sequences, validation_results = create_sequences_lopo(test_patient_id)
                
                if not sequences:
                    print(f"❌ No sequences generated for test patient {test_patient_id}!")
                else:
                    # Count splits
                    train_count = sum(1 for s in sequences if s['split'] == 'train')
                    test_count = sum(1 for s in sequences if s['split'] == 'test')
                    
                    print(f"\nSequence Summary:")
                    print(f"  - Training sequences: {train_count}")
                    print(f"  - Test sequences: {test_count}")
                    print(f"  - Total sequences: {len(sequences)}")
                    
                    # Save to file
                    output_filename = f"lopo_test_{test_patient_id}_sequences_{TASK_MODE}.json"
                    split_counts = {
                        'train': train_count,
                        'test': test_count,
                        'total': len(sequences)
                    }
                    
                    save_sequences_to_file(
                        sequences,
                        validation_results,
                        output_file=output_filename,
                        split_summary=split_counts,
                    )
                    
                    print(f"✅ Saved to {output_filename}")
            
            except Exception as e:
                print(f"❌ Error processing patient {test_patient_id}: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
