#!/usr/bin/env python3
"""
Calculate per-seizure accuracy and false positive rate.
"""
import numpy as np
import h5py


def apply_smoothing(all_predictions, t, x):
    """
    Apply smoothing to a list of predictions.

    Args:
        all_predictions: List of predictions.
        t: Window size for smoothing.
        x: Minimum number of positive predictions in the window.

    Returns:
        List of smoothed predictions.
    """
    smoothed_predictions = []
    for i in range(len(all_predictions) - t + 1):
        window = all_predictions[i : i + t]
        if sum(window) >= x:
            smoothed_predictions.append(1)
        else:
            smoothed_predictions.append(0)
    return smoothed_predictions


def calculate_per_seizure_accuracy(
    true_labels,
    raw_predictions,
    test_dataset,
    sequence_stride_seconds,
    window_size,
    threshold_x,
    refractory_period_seconds=1800,  # 30 minutes default
):
    """
    Calculates per-seizure accuracy and false positive rate on chronologically sorted data.

    Args:
        true_labels (list or np.array): Ground truth labels for each sequence.
        raw_predictions (list or np.array): Raw model predictions (0s and 1s).
        test_dataset (EEGDataset): The test dataset object to access HDF5 metadata.
        sequence_stride_seconds (int): The time step between consecutive sequences (stride).
        window_size (int): The 't' parameter for smoothing (window size).
        threshold_x (int): The 'x' parameter for smoothing (count threshold).
        refractory_period_seconds (int): Duration to ignore subsequent alarms after a false positive.

    Returns:
        dict: A dictionary containing the per-seizure accuracy and false positive rate.
    """
    # --- 1. Chronological Sorting ---
    try:
        with h5py.File(test_dataset.h5_file_path, "r") as f:
            filenames = [n.decode("utf-8") for n in f["segment_info/file_names"][:]]
            start_times = f["segment_info/start_times"][:]

            if len(filenames) != len(true_labels):
                print(
                    f"⚠️ Metadata length mismatch! Data: {len(true_labels)}, Meta: {len(filenames)}"
                )
                sorted_indices = range(len(true_labels))
            else:
                meta_list = list(zip(range(len(filenames)), filenames, start_times))
                meta_list.sort(key=lambda x: (x[1], x[2]))
                sorted_indices = [x[0] for x in meta_list]
    except Exception as e:
        print(f"⚠️ Could not load metadata for sorting: {e}")
        sorted_indices = range(len(true_labels))

    # Apply sorting to get chronological labels and predictions
    sorted_labels = np.array(true_labels)[sorted_indices]
    sorted_raw_predictions = np.array(raw_predictions)[sorted_indices]

    # --- 2. Smoothing on Sorted Data ---
    sorted_smoothed_predictions = apply_smoothing(
        sorted_raw_predictions, window_size, threshold_x
    )

    # The smoothed predictions are shorter. We need to align the labels.
    offset = window_size - 1
    aligned_labels = sorted_labels[offset:]

    if len(aligned_labels) != len(sorted_smoothed_predictions):
        print(
            f"⚠️ Post-smoothing length mismatch! Aligned Labels: {len(aligned_labels)}, Smoothed Preds: {len(sorted_smoothed_predictions)}"
        )
        min_len = min(len(aligned_labels), len(sorted_smoothed_predictions))
        aligned_labels = aligned_labels[:min_len]
        sorted_smoothed_predictions = sorted_smoothed_predictions[:min_len]

    # --- 3. Per-Seizure Accuracy Calculation ---
    # Identify seizure blocks in the ORIGINAL (sorted) labels
    # We use the original labels to find boundaries, then map to smoothed indices
    is_preictal = (sorted_labels == 1).astype(int)
    diff = np.diff(np.concatenate(([0], is_preictal, [0])))
    seizure_starts = np.where(diff == 1)[0]
    seizure_ends = np.where(diff == -1)[0]

    num_seizures = len(seizure_starts)
    correctly_predicted_seizures = 0

    if num_seizures > 0:
        for i, (start_idx, end_idx) in enumerate(zip(seizure_starts, seizure_ends)):
            # Adjust indices for the smoothing offset
            # smoothed_pred[0] corresponds to raw_preds[0...t-1]
            # So if seizure starts at index S in raw, the first window CONTAINING it ends at S+t-1.
            # But here we just want to see if *any* window covering the seizure block triggered.
            # Simple alignment: smoothed index i corresponds to raw time i + offset
            # So raw index j maps to smoothed index j - offset
            smoothed_start = max(0, start_idx - offset)
            smoothed_end = max(0, end_idx - offset)

            if smoothed_start < smoothed_end and smoothed_end <= len(
                sorted_smoothed_predictions
            ):
                prediction_slice = sorted_smoothed_predictions[
                    smoothed_start:smoothed_end
                ]
                if np.sum(prediction_slice) > 0:
                    correctly_predicted_seizures += 1

        per_seizure_accuracy = correctly_predicted_seizures / num_seizures
    else:
        per_seizure_accuracy = 1.0

    # --- 4. False Positive Rate (FPR/h) with Refractory Period ---
    false_positive_events = 0
    refractory_steps = int(refractory_period_seconds / sequence_stride_seconds)

    # Convert to array for easier indexing
    preds = np.array(sorted_smoothed_predictions)
    labels = np.array(aligned_labels)

    i = 0
    while i < len(preds):
        # Check if current point is Interictal (Label 0) AND Prediction is 1
        if labels[i] == 0:
            if preds[i] == 1:
                # Found a False Positive Event
                false_positive_events += 1
                # Skip ahead by refractory period
                i += refractory_steps
                continue
        i += 1

    # Total Interictal Duration
    num_interictal_windows = np.sum(labels == 0)
    total_interictal_duration_hours = (
        num_interictal_windows * sequence_stride_seconds
    ) / 3600

    if total_interictal_duration_hours > 0:
        fp_rate_per_hour = false_positive_events / total_interictal_duration_hours
    else:
        fp_rate_per_hour = 0.0

    print(f"DEBUG: Stride={sequence_stride_seconds}s, RefractorySteps={refractory_steps}")
    print(f"DEBUG: Duration={total_interictal_duration_hours:.2f}h, FPEvents={false_positive_events}, FPR={fp_rate_per_hour:.2f}")

    return {
        "per_seizure_accuracy": per_seizure_accuracy,
        "fp_rate_per_hour": fp_rate_per_hour,
        "total_seizures": num_seizures,
        "correctly_predicted_seizures": correctly_predicted_seizures,
        "total_independent_fp_events": false_positive_events,
        "total_interictal_duration_hours": total_interictal_duration_hours,
    }
