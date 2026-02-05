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
    LOPO_FOLD_ID,
    MAX_HORIZON_SEC,
    POST_ICTAL_EXCLUSION_SEC,
    TRAIN_TEST_SPLIT_RATIO,
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
    """Create sequences of consecutive segments with TTS Regression labels.

    Logic:
    - Uniform stride (non-overlapping): stride = sequence_duration.
    - Labels: Time until NEXT seizure start, capped at MAX_HORIZON_SEC.
    - Exclusion: Discard data during seizures or within POST_ICTAL_EXCLUSION_SEC after.
    """
    sequences = []
    sequence_duration = SEGMENT_DURATION * SEQUENCE_LENGTH
    stride = sequence_duration  # Force 0% overlap for all sequences

    # Generate sequence start times
    sequence_start_local = SAFETY_MARGIN
    while sequence_start_local + sequence_duration + SAFETY_MARGIN <= file_duration:
        # Convert to global timeline
        sequence_start_global = file_offset + sequence_start_local
        sequence_end_global = sequence_start_global + sequence_duration

        # 1. Identify "Next" and "Previous" Seizures
        next_seizure = None
        prev_seizure = None
        is_ictal = False

        for seizure in all_seizures_global:
            s_start = seizure["start_sec_global"]
            s_end = seizure["end_sec_global"]

            # Check if current sequence overlaps with this seizure (Ictal)
            if sequence_start_global < s_end and sequence_end_global > s_start:
                is_ictal = True
                break

            # Find next seizure
            if s_start >= sequence_end_global:
                if next_seizure is None or s_start < next_seizure["start_sec_global"]:
                    next_seizure = seizure

            # Find previous seizure
            if s_end <= sequence_start_global:
                if prev_seizure is None or s_end > prev_seizure["end_sec_global"]:
                    prev_seizure = seizure

        # 2. Apply Exclusion Rules
        # Rule A: Discard Ictal data
        if is_ictal:
            sequence_start_local += stride
            continue

        # Rule B: Discard Post-Ictal data (1 hour after previous seizure)
        if prev_seizure:
            time_since_prev = sequence_start_global - prev_seizure["end_sec_global"]
            if time_since_prev < POST_ICTAL_EXCLUSION_SEC:
                sequence_start_local += stride
                continue

        # Rule C: If no next seizure exists in the record, we can't label it properly
        # (Alternatively, we could label it as MAX_HORIZON, but exclusion is safer)
        if next_seizure is None:
            sequence_start_local += stride
            continue

        # 3. Calculate TTS Label (Sawtooth)
        # Use the end of the sequence as the reference point for countdown
        tts_seconds = next_seizure["start_sec_global"] - sequence_end_global
        capped_tts = min(tts_seconds, MAX_HORIZON_SEC)

        # 4. Final Sequence Metadata
        segment_starts = [
            sequence_start_local + (i * SEGMENT_DURATION)
            for i in range(SEQUENCE_LENGTH)
        ]

        sequence = {
            "patient_id": str(patient_id),
            "file": str(filename),
            "sequence_start_sec": float(sequence_start_local),
            "sequence_end_sec": float(sequence_start_local + sequence_duration),
            "sequence_duration_sec": float(sequence_duration),
            "segment_starts": [float(s) for s in segment_starts],
            "num_segments": int(SEQUENCE_LENGTH),
            "type": "tts_regression",
            "tts_label": float(capped_tts),  # The target for regression
            "is_preictal": bool(tts_seconds < PREICTAL_WINDOW),  # Standard python bool
            "global_start_sec": float(sequence_start_global),
            "next_seizure_id": int(next_seizure["seizure_id"]),
        }
        sequences.append(sequence)
        sequence_start_local += stride

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
    """Assign sequences to train/test splits by GROUPING by seizure.

    Logic:
    - Group all sequences by their 'next_seizure_id'.
    - Randomly shuffle the seizure groups.
    - Assign ~70% of seizures (and their sequences) to Train.
    - Assign remaining seizures to Test.
    - This prevents temporal leakage between train/test for the same countdown.
    """
    if not sequences:
        return [], {}, {}

    import random
    rng = random.Random(random_seed)

    # 1. Group sequences by seizure_id
    seizure_groups = {}
    for seq in sequences:
        sid = seq.get("next_seizure_id", -1)
        if sid not in seizure_groups:
            seizure_groups[sid] = []
        seizure_groups[sid].append(seq)

    group_ids = list(seizure_groups.keys())
    rng.shuffle(group_ids)

    n_groups = len(group_ids)
    
    # Fallback for patients with very few seizures
    if n_groups < 2:
        print(f"[WARNING] Patient {patient_id} has only {n_groups} seizure(s). "
              "Falling back to random sequence split (LEAKAGE RISK).")
        rng.shuffle(sequences)
        n_train = int(len(sequences) * TRAIN_TEST_SPLIT_RATIO)
        train_sequences = sequences[:n_train]
        test_sequences = sequences[n_train:]
    else:
        # Split groups
        n_train_groups = max(1, int(n_groups * TRAIN_TEST_SPLIT_RATIO))
        # Ensure at least one group in test if possible
        if n_train_groups == n_groups and n_groups > 1:
            n_train_groups = n_groups - 1
            
        train_group_ids = group_ids[:n_train_groups]
        test_group_ids = group_ids[n_train_groups:]
        
        train_sequences = []
        for gid in train_group_ids:
            train_sequences.extend(seizure_groups[gid])
            
        test_sequences = []
        for gid in test_group_ids:
            test_sequences.extend(seizure_groups[gid])

        print(f"  Split by Seizure: {len(train_group_ids)} seizures in Train, {len(test_group_ids)} in Test")

    # 3. Label
    for seq in train_sequences:
        seq["split"] = "train"
    for seq in test_sequences:
        seq["split"] = "test"

    retained_sequences = train_sequences + test_sequences

    # 4. Compute counts for summary
    import numpy as np
    split_counts = {
        "train": {
            "total": len(train_sequences),
            "mean_tts": float(np.mean([s["tts_label"] for s in train_sequences]))
            if train_sequences
            else 0.0,
        },
        "test": {
            "total": len(test_sequences),
            "mean_tts": float(np.mean([s["tts_label"] for s in test_sequences]))
            if test_sequences
            else 0.0,
        },
    }

    return retained_sequences, split_counts, {}


def save_sequences_to_file(
    sequences,
    validation_results,
    output_file=None,
    balancing_stats=None,
    split_summary=None,
    extra_summary=None,
):
    """Save all sequences to a single file with TTS Regression information."""

    if output_file is None:
        raise ValueError("output_file must be specified")

    # Get validation summary
    validation_summary = (
        get_validation_summary(validation_results) if validation_results else {}
    )

    data = {
        "sequences": sequences,
        "summary": {
            "task_mode": "tts_regression",
            "total_sequences": len(sequences),
            "patients_processed": len(set([s["patient_id"] for s in sequences])),
            "max_horizon_sec": MAX_HORIZON_SEC,
            "post_ictal_exclusion_sec": POST_ICTAL_EXCLUSION_SEC,
            "segment_duration": SEGMENT_DURATION,
            "sequence_length": SEQUENCE_LENGTH,
            "sequence_total_duration": SEGMENT_DURATION * SEQUENCE_LENGTH,
        },
        "validation_info": validation_summary,
    }

    if split_summary:
        data["summary"]["per_split"] = split_summary

    if extra_summary:
        data["summary"].update(extra_summary)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSequences saved to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")


def print_summary(sequences):
    """Print comprehensive summary for TTS Regression"""
    import numpy as np
    
    print(f"\n=== OVERALL SUMMARY (TTS REGRESSION) ===")
    print(f"Total sequences: {len(sequences)}")
    print(f"Max Horizon: {MAX_HORIZON_SEC/3600:.1f} hours")
    print(f"Post-Ictal Exclusion: {POST_ICTAL_EXCLUSION_SEC/60:.1f} min")

    # Per-split breakdown
    splits = sorted(list(set([s.get("split", "none") for s in sequences])))
    for split_name in splits:
        split_seqs = [s for s in sequences if s.get("split") == split_name]
        mean_tts = np.mean([s["tts_label"] for s in split_seqs])
        print(
            f"Split '{split_name}': {len(split_seqs)} sequences (Mean TTS: {mean_tts/60:.1f} min)"
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
            for split_name in split_counts:
                counts = split_counts.get(split_name, {})
                print(
                    f"  {split_name}: {counts.get('total', 0)} sequences "
                    f"({counts.get(positive_label, 0)} {positive_label}, "
                    f"{counts.get('interictal', 0)} interictal)"
                )

            # Print balance stats
            if balance_stats:
                print(f"\n=== BALANCING SUMMARY ===")
                for split_name in split_counts:
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
