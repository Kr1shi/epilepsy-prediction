import sys
import os

# Add project root to sys.path to allow imports from top-level modules
sys.path.append(os.path.join(os.getcwd()))

import act_gpu as act_lib
import numpy as np
import cupy as cp
import pandas as pd
import mne
import csv
import time
from cupy.cuda import stream, memory
import monitoringclass
from data_segmentation_helpers.config import TARGET_CHANNELS

start_time = time.time()

# Enable Unified Memory to allow dynamic memory management                                 
cp.cuda.set_allocator(memory.malloc_managed)


# Initialize monitoring class
monitoring = monitoringclass.MonitoringClass()
monitor = True

epoch = 5
act = act_lib.ACT(
    FS=256,
    length=epoch * 256,
    tc_info=(0, epoch * 256, 64),
    fc_info=(0.6, 15, 1),
    logDt_info=(-4, 0, 0.3),
    c_info=(-10, 10, 0.75),
    force_regenerate=True,
    mute=False,
    monitor = monitor
)

def select_target_channels(raw):
    """Select target channels with fuzzy matching (from data_preprocessing.py)"""
    available_channels = []
    clean_channel_names = []
    for ch in TARGET_CHANNELS:
        if ch in raw.ch_names:
            available_channels.append(ch)
            clean_channel_names.append(ch)
        else:
            matches = [n for n in raw.ch_names if n.startswith(ch + "-")]
            if matches:
                available_channels.append(matches[0])
                clean_channel_names.append(ch)
    
    # Pick channels and rename them to standard names for consistency
    raw_selected = raw.copy().pick_channels(available_channels)
    # Create mapping from actual name to clean name
    rename_map = {actual: clean for actual, clean in zip(available_channels, clean_channel_names)}
    raw_selected.rename_channels(rename_map)
    
    return raw_selected, clean_channel_names

# Load EEG data
data_file = os.path.join(os.getcwd(), "./physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf")
raw_data = mne.io.read_raw_edf(data_file, preload=True, verbose=False)

# Use robust channel selection
raw_data, channel_names = select_target_channels(raw_data)

raw_data.notch_filter(freqs=60, fir_design="firwin", verbose=False)

eeg_data = raw_data.get_data().T  # Shape (samples, channels)
eeg_data_gpu = cp.asarray(eeg_data, dtype=cp.float32) 


epoch_length = epoch * act.FS

# num_epochs = eeg_data.shape[0] // epoch_length # When doing the entire dataset
num_epochs = 1
output_csv = "act_results_sub-1_optimized.csv"

with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Params", "Coeffs", "Error", "Residue"])
    
    for epoch_idx in range(num_epochs):
        
        if monitor:
            monitoring.start_CPU_monitoring()
            monitoring.start_GPU_monitoring()

        start_idx = epoch_idx * epoch_length
        end_idx = start_idx + epoch_length
        print(f"Processing epoch {epoch_idx + 1}/{num_epochs}")

        for electrode_idx, electrode_name in enumerate(raw_data.ch_names):
            segment_gpu = eeg_data_gpu[start_idx:end_idx, electrode_idx]

            # Use a stream for parallelism, but make sure to synchronize
            my_stream = cp.cuda.Stream()
            with my_stream:
                result = act.transform(segment_gpu, order=6, debug=False)
            my_stream.synchronize()  # Make sure the GPU has completed work

            # Move results to CPU
            params = cp.asnumpy(result["params"]).tolist()
            coeffs = cp.asnumpy(result["coeffs"]).tolist()
            residue = cp.asnumpy(result["residue"]).tolist()

            writer.writerow([epoch_idx+1, params, coeffs, result["error"], residue])
        if monitor:
            monitoring.stop_CPU_monitoring()
            monitoring.stop_GPU_monitoring()

                
    print(f"Epoch {epoch_idx} completed in {time.time() - start_time:.2f} sec")

print(f"Total processing time: {time.time() - start_time:.2f} sec")
