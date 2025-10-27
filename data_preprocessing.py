import mne
import numpy as np
import h5py
import json
import os
import logging
import warnings
import random
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from scipy.signal import spectrogram
from multiprocessing import Pool, cpu_count
from functools import partial
from data_segmentation_helpers.config import *

# Suppress MNE warnings that don't affect processing
warnings.filterwarnings("ignore", message="Channel names are not unique")
warnings.filterwarnings("ignore", message=".*duplicates.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Scaling factor is not defined")
warnings.filterwarnings("ignore", message=".*scaling factor.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Number of records from the header does not match")
warnings.filterwarnings("ignore", message=".*file size.*", category=RuntimeWarning)

# Set MNE log level early
mne.set_log_level('ERROR')

class EEGPreprocessor:
    def __init__(self, input_json_path: str = None):
        # Auto-generate input filename based on task mode if not provided
        if input_json_path is None:
            input_json_path = f"{OUTPUT_PREFIX}_sequences_{TASK_MODE}.json"
        self.input_json_path = input_json_path
        self.output_dir = Path("preprocessing")
        self.dataset_prefix = OUTPUT_PREFIX
        self.data_dir = self.output_dir / "data" / self.dataset_prefix
        self.logs_dir = self.output_dir / "logs" / self.dataset_prefix
        self.checkpoint_dir = self.output_dir / "checkpoints" / self.dataset_prefix
        self.checkpoint_file = self.checkpoint_dir / "progress.json"
        self.balance_seed = SINGLE_PATIENT_RANDOM_SEED if SINGLE_PATIENT_MODE else 42

        # Processing settings
        self.checkpoint_interval = 50
        self.train_ratio = 0.70
        self.val_ratio = 0.15
        self.test_ratio = 0.15

        # Statistics tracking
        self.removed_segments = {
            'wrong_duration': 0,
            'beyond_file_bounds': 0,
            'processing_errors': 0
        }

        # Normalization statistics (computed from training data only)
        self.norm_stats_file = self.checkpoint_dir / 'normalization_stats.json'
        self.global_mean = None
        self.global_std = None

        # Initialize logging and directories
        self.setup_logging_and_directories()

        # Load checkpoint if exists
        self.checkpoint = self.load_checkpoint()

        self.logger.info("EEG Preprocessor initialized")
        self.logger.info(f"Target channels: {TARGET_CHANNELS}")
        self.logger.info(f"Filter settings: {LOW_FREQ_HZ}-{HIGH_FREQ_HZ} Hz, notch: {NOTCH_FREQ_HZ} Hz")
        self.logger.info(f"Segment duration: {SEGMENT_DURATION} seconds")
        self.logger.info("Note: Channel validation performed during segmentation phase")

        # Multiprocessing setup
        available_cores = cpu_count()
        self.n_workers = min(PREPROCESSING_WORKERS, available_cores)
        self.logger.info(f"Multiprocessing: Using {self.n_workers}/{available_cores} CPU cores for parallel preprocessing")
        self.logger.info(f"MNE parallel filtering: {MNE_N_JOBS} jobs per sequence")

    @property
    def positive_label(self):
        """Get positive class label based on task mode"""
        return 'preictal' if TASK_MODE == 'prediction' else 'ictal'

    @staticmethod
    def group_sequences_by_file(sequences: List[Dict]) -> Dict[Tuple[str, str], List[Dict]]:
        """Group sequences by (patient_id, filename) for efficient batch processing.

        Args:
            sequences: List of sequence dictionaries

        Returns:
            Dictionary mapping (patient_id, filename) to list of sequences from that file
        """
        from collections import defaultdict
        file_groups = defaultdict(list)

        for sequence in sequences:
            key = (sequence['patient_id'], sequence['file'])
            file_groups[key].append(sequence)

        return dict(file_groups)

    def setup_logging_and_directories(self):
        """Setup logging and create directory structure"""
        # Create directories
        for dir_path in [self.data_dir, self.logs_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.logs_dir / "preprocessing.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*60)
        self.logger.info("EEG PREPROCESSING PIPELINE STARTED")
        self.logger.info("="*60)
        self.logger.info(f"Output prefix: {self.dataset_prefix}")

    def load_checkpoint(self) -> Dict:
        """Load checkpoint if exists, handle corrupted files"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                
                # Convert processed_segments list back to set for efficient lookups
                if isinstance(checkpoint.get('processed_segments'), list):
                    checkpoint['processed_segments'] = set(checkpoint['processed_segments'])
                
                self.logger.info(f"Loaded checkpoint: {len(checkpoint.get('processed_segments', []))} segments processed")
                return checkpoint
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                self.logger.warning(f"Corrupted checkpoint file detected: {e}")
                self.logger.warning("Starting fresh - deleting corrupted checkpoint")
                # Backup the corrupted file
                backup_path = self.checkpoint_file.with_suffix('.json.corrupted')
                self.checkpoint_file.rename(backup_path)
                self.logger.info(f"Corrupted checkpoint backed up to: {backup_path}")
        
        # Return fresh checkpoint
        return {
            "processed_segments": set(),  # Use set for O(1) lookup
            "current_split": None,
            "splits_completed": [],
            "total_segments": 0,
            "processed_count": 0,
            "start_time": datetime.now().isoformat(),
            "valid_segments": {}
        }

    def save_checkpoint(self):
        """Save current progress, handling non-JSON serializable objects"""
        # Convert sets to lists for JSON serialization
        checkpoint_copy = self.checkpoint.copy()
        
        # Convert processed_segments set to list for JSON
        if isinstance(checkpoint_copy.get('processed_segments'), set):
            checkpoint_copy['processed_segments'] = list(checkpoint_copy['processed_segments'])
        
        # Handle validation stats sets
        if 'valid_segments' in checkpoint_copy and 'stats' in checkpoint_copy['valid_segments']:
            stats = checkpoint_copy['valid_segments']['stats']
            if 'patients_processed' in stats and isinstance(stats['patients_processed'], set):
                stats['patients_processed'] = list(stats['patients_processed'])
            if 'patients_with_no_valid' in stats and isinstance(stats['patients_with_no_valid'], set):
                stats['patients_with_no_valid'] = list(stats['patients_with_no_valid'])
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_copy, f, indent=2)

    def compute_normalization_stats(self, train_spectrograms):
        """Compute global mean and std from training spectrograms

        Args:
            train_spectrograms: List of spectrogram arrays from training data
        """
        self.logger.info("Computing normalization statistics from training data...")

        # Stack all spectrograms
        all_specs = np.concatenate(train_spectrograms, axis=0)

        # Compute global statistics
        self.global_mean = np.mean(all_specs)
        self.global_std = np.std(all_specs)

        # Save to file
        stats = {
            'global_mean': float(self.global_mean),
            'global_std': float(self.global_std),
            'n_samples': len(train_spectrograms),
            'total_values': all_specs.size
        }

        with open(self.norm_stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"Normalization stats computed:")
        self.logger.info(f"  Mean: {self.global_mean:.6f}")
        self.logger.info(f"  Std: {self.global_std:.6f}")
        self.logger.info(f"  Saved to: {self.norm_stats_file}")

    def load_normalization_stats(self):
        """Load normalization statistics from file"""
        if self.norm_stats_file.exists():
            with open(self.norm_stats_file, 'r') as f:
                stats = json.load(f)

            self.global_mean = stats['global_mean']
            self.global_std = stats['global_std']

            self.logger.info("Loaded normalization statistics:")
            self.logger.info(f"  Mean: {self.global_mean:.6f}")
            self.logger.info(f"  Std: {self.global_std:.6f}")
            return True
        else:
            self.logger.warning("No normalization statistics file found")
            return False

    # Channel validation has been moved to data_segmentation.py
    # This ensures we fail fast and only create sequences for files with valid channels
    # The input JSON file already contains only validated sequences

    def balance_and_split_data(self, sequences: List[Dict]) -> Dict[str, List[Dict]]:
        """Patient-level split for sequences (no balancing to preserve patient distribution)"""
        self.logger.info("Performing patient-level split for sequences...")

        # Separate by class (using dynamic positive label)
        positive_sequences = [s for s in sequences if s['type'] == self.positive_label]
        interictal_sequences = [s for s in sequences if s['type'] == 'interictal']

        self.logger.info(f"Total sequences: {len(positive_sequences)} {self.positive_label}, {len(interictal_sequences)} interictal")

        # Get unique patients
        all_patients = sorted(set([s['patient_id'] for s in sequences]))
        self.logger.info(f"Total patients: {len(all_patients)}")

        # Split patients (not sequences) into train/val/test
        rng = np.random.RandomState(42)
        rng.shuffle(all_patients)

        n_train = int(len(all_patients) * self.train_ratio)
        n_val = int(len(all_patients) * self.val_ratio)

        train_patients = set(all_patients[:n_train])
        val_patients = set(all_patients[n_train:n_train+n_val])
        test_patients = set(all_patients[n_train+n_val:])

        self.logger.info(f"Patient split: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test")
        self.logger.info(f"Train patients: {sorted(train_patients)}")
        self.logger.info(f"Val patients: {sorted(val_patients)}")
        self.logger.info(f"Test patients: {sorted(test_patients)}")

        # Assign sequences to splits based on patient
        splits = {
            'train': [s for s in sequences if s['patient_id'] in train_patients],
            'val': [s for s in sequences if s['patient_id'] in val_patients],
            'test': [s for s in sequences if s['patient_id'] in test_patients]
        }

        # Log split statistics
        for split_name, split_seqs in splits.items():
            positive_count = sum(1 for s in split_seqs if s['type'] == self.positive_label)
            interictal_count = len(split_seqs) - positive_count
            patients = set([s['patient_id'] for s in split_seqs])
            self.logger.info(f"{split_name}: {len(split_seqs)} sequences ({positive_count} {self.positive_label}, {interictal_count} interictal) from {len(patients)} patients")

        return splits

    def balance_split_sequences(self, splits: Dict[str, List[Dict]]) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict[str, int]]]:
        """Downsample majority class within each split to enforce class balance.

        Args:
            splits: Dict mapping split name to list of sequence dictionaries.

        Returns:
            Tuple of (balanced_splits, balance_stats)
        """
        rng = random.Random(self.balance_seed)
        balanced_splits = {}
        balance_stats = {}

        for split_name, split_sequences in splits.items():
            positives = [seq for seq in split_sequences if seq['type'] == self.positive_label]
            interictals = [seq for seq in split_sequences if seq['type'] == 'interictal']
            others = [seq for seq in split_sequences if seq['type'] not in (self.positive_label, 'interictal')]

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

            dropped_positive = len(positives) - target_count
            dropped_interictal = len(interictals) - target_count

            combined = kept_positive + kept_interictal
            rng.shuffle(combined)

            if others:
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

    def select_target_channels(self, raw, target_channels=TARGET_CHANNELS):
        """Select target channels, handling duplicates"""
        available_channels = []
        clean_channel_names = []
        
        for ch in target_channels:
            if ch in raw.ch_names:
                available_channels.append(ch)
                clean_channel_names.append(ch)
            else:
                # Check for renamed duplicates
                duplicate_matches = [name for name in raw.ch_names 
                                   if name.startswith(ch + '-') and name.split('-')[-1].isdigit()]
                if duplicate_matches:
                    available_channels.append(duplicate_matches[0])
                    clean_channel_names.append(ch)  # Use clean name
        
        raw_filtered = raw.copy().pick(available_channels)
        return raw_filtered, clean_channel_names

    def remove_amplitude_artifacts(self, raw):
        """Remove extreme amplitude artifacts"""
        data = raw.get_data()
        artifact_mask = np.zeros_like(data, dtype=bool)
        
        for ch in range(data.shape[0]):
            channel_data = data[ch, :]
            median_val = np.median(channel_data)
            mad = np.median(np.abs(channel_data - median_val))
            threshold = ARTIFACT_THRESHOLD_STD * mad * 1.4826
            
            artifact_mask[ch, :] = np.abs(channel_data - median_val) > threshold
        
        # Replace artifacts with interpolated values
        for ch in range(data.shape[0]):
            if np.any(artifact_mask[ch, :]):
                artifact_indices = np.where(artifact_mask[ch, :])[0]
                clean_indices = np.where(~artifact_mask[ch, :])[0]
                
                if len(clean_indices) > 10:
                    data[ch, artifact_indices] = np.interp(
                        artifact_indices, clean_indices, data[ch, clean_indices]
                    )
        
        raw._data = data
        return np.sum(artifact_mask, axis=1)

    def robust_normalize(self, data):
        """Apply robust normalization"""
        normalized_data = np.zeros_like(data)
        scaler = RobustScaler()
        
        for ch in range(data.shape[0]):
            normalized_data[ch, :] = scaler.fit_transform(
                data[ch, :].reshape(-1, 1)
            ).flatten()
        
        return normalized_data

    def apply_stft(self, data, sampling_rate):
        """Apply STFT to create spectrograms

        Returns:
            stft_spectrograms: Complex STFT coefficients (n_channels, n_freqs, n_times)
            frequencies: Frequency values for each bin
            time_array: Time values for each STFT bin (centers of windows)
        """
        spectrograms = []
        time_array = None  # Same for all channels

        for ch in range(data.shape[0]):
            f, t, Zxx = spectrogram(
                data[ch, :],
                fs=sampling_rate,
                nperseg=STFT_NPERSEG,
                noverlap=STFT_NOVERLAP,
                window='hann'
            )

            # Store time array (same for all channels)
            if time_array is None:
                time_array = t

            # Filter to desired frequency range
            freq_mask = (f >= LOW_FREQ_HZ) & (f <= HIGH_FREQ_HZ)
            filtered_spec = Zxx[freq_mask, :]
            spectrograms.append(filtered_spec)

        stft_spectrograms = np.array(spectrograms)
        frequencies = f[freq_mask]

        return stft_spectrograms, frequencies, time_array

    def preprocess_sequences_from_file(self, patient_id: str, filename: str, sequences: List[Dict]) -> List[Optional[Dict]]:
        """Process multiple sequences from the same EDF file efficiently.

        This method reads and filters the file ONCE, computes STFT ONCE on the entire range,
        then extracts all sequences by slicing the pre-computed spectrogram.
        Major optimization: reduces redundant file I/O, filtering, and STFT operations.

        Args:
            patient_id: Patient identifier
            filename: EDF filename
            sequences: List of sequence dictionaries from this file

        Returns:
            List of processed sequence dictionaries (or None for failed sequences)
        """
        try:
            edf_path = f"physionet.org/files/chbmit/1.0.0/{patient_id}/{filename}"

            # Read EDF file ONCE for all sequences
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            sampling_rate = raw.info['sfreq']

            # Select target channels
            raw_selected, clean_channel_names = self.select_target_channels(raw)

            # Find global min/max time range across ALL sequences from this file
            all_segment_starts = []
            for seq in sequences:
                all_segment_starts.extend(seq['segment_starts'])

            min_time = min(all_segment_starts)
            max_time = max(all_segment_starts) + SEGMENT_DURATION

            # Add padding for filter edge effects
            padding = 5.0
            crop_tmin = max(0, min_time - padding)
            crop_tmax = min(raw_selected.times[-1], max_time + padding)

            # Crop to the range needed by ALL sequences (with padding)
            raw_cropped = raw_selected.copy().crop(tmin=crop_tmin, tmax=crop_tmax)

            # Filter ONCE for all sequences from this file
            raw_filtered = raw_cropped.copy()
            raw_filtered.filter(l_freq=LOW_FREQ_HZ, h_freq=HIGH_FREQ_HZ, fir_design='firwin',
                              n_jobs=MNE_N_JOBS, verbose=False)
            raw_filtered.notch_filter(freqs=NOTCH_FREQ_HZ, n_jobs=MNE_N_JOBS, verbose=False)

            # ✅ OPTIMIZATION: Artifact removal on ENTIRE filtered range (more robust statistics)
            self.remove_amplitude_artifacts(raw_filtered)

            # ✅ OPTIMIZATION: Convert to microvolts ONCE
            # MNE reads EEG in Volts (~0.0002V), but |STFT|² on such small values
            # produces numerical zeros. Scaling to μV (~200μV) prevents this.
            filtered_data_uv = raw_filtered.get_data() * 1e6

            # ✅ OPTIMIZATION: Compute STFT ONCE on entire range
            stft_coeffs, frequencies, time_array = self.apply_stft(filtered_data_uv, sampling_rate)

            # ✅ OPTIMIZATION: Compute power spectrogram ONCE
            full_power_spectrogram = np.abs(stft_coeffs) ** 2

            # Now extract all sequences by slicing the pre-computed spectrogram
            processed_sequences = []

            for sequence in sequences:
                try:
                    sequence_spectrograms = []

                    # Extract each segment in this sequence by slicing the spectrogram
                    for segment_start in sequence['segment_starts']:
                        # Calculate segment boundaries relative to crop start
                        segment_start_relative = segment_start - crop_tmin
                        segment_end_relative = segment_start_relative + SEGMENT_DURATION

                        # Find STFT time bins within this segment
                        # Segments start at integer multiples of 5s, STFT bins at 0.5s intervals
                        # → perfect alignment, no approximation needed!
                        bin_mask = (time_array >= segment_start_relative) & (time_array < segment_end_relative)

                        # Slice pre-computed power spectrogram (channels × freqs × time)
                        power_spectrogram = full_power_spectrogram[:, :, bin_mask]

                        # Apply log transform
                        if APPLY_LOG_TRANSFORM:
                            power_spectrogram = np.log10(power_spectrogram + LOG_TRANSFORM_EPSILON)

                        # Apply normalization (using training statistics)
                        if self.global_mean is not None and self.global_std is not None:
                            power_spectrogram = (power_spectrogram - self.global_mean) / (self.global_std + 1e-8)

                        sequence_spectrograms.append(power_spectrogram)

                    # Stack into sequence: (seq_len, n_channels, n_freqs, n_times)
                    sequence_array = np.stack(sequence_spectrograms, axis=0)

                    processed_sequences.append({
                        'spectrogram': sequence_array,
                        'label': 1 if sequence['type'] == self.positive_label else 0,
                        'patient_id': sequence['patient_id'],
                        'sequence_start_sec': sequence['sequence_start_sec'],
                        'sequence_end_sec': sequence['sequence_end_sec'],
                        'file_name': sequence['file'],
                        'time_to_seizure': sequence.get('time_to_seizure', -1),
                        'channel_names': clean_channel_names,
                        'frequencies': frequencies
                    })

                except Exception as e:
                    self.logger.error(f"Failed to process sequence from {patient_id}/{filename} at {sequence['sequence_start_sec']}s: {e}")
                    self.removed_segments['processing_errors'] += 1
                    processed_sequences.append(None)

            return processed_sequences

        except Exception as e:
            self.logger.error(f"Failed to process file {patient_id}/{filename}: {e}")
            # Return None for all sequences if file processing fails
            return [None] * len(sequences)

    def preprocess_sequence(self, sequence: Dict) -> Optional[Dict]:
        """Preprocess an entire sequence (multiple consecutive segments)"""
        try:
            # Process each segment in the sequence
            sequence_spectrograms = []
            edf_path = f"physionet.org/files/chbmit/1.0.0/{sequence['patient_id']}/{sequence['file']}"

            # Read EDF file once for the whole sequence
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            sampling_rate = raw.info['sfreq']

            # Select target channels
            raw_selected, clean_channel_names = self.select_target_channels(raw)

            # ✅ OPTIMIZATION 1: Crop to sequence range BEFORE filtering
            # Only filter the portion of the file we actually need
            min_time = min(sequence['segment_starts'])
            max_time = max(sequence['segment_starts']) + SEGMENT_DURATION

            # Add padding for filter edge effects (5 seconds on each side)
            padding = 5.0
            crop_tmin = max(0, min_time - padding)
            crop_tmax = min(raw_selected.times[-1], max_time + padding)

            # Crop to the sequence range (with padding)
            raw_cropped = raw_selected.copy().crop(tmin=crop_tmin, tmax=crop_tmax)

            # ✅ OPTIMIZATION 2: Filter ONCE (not per segment!) and only the cropped range
            raw_filtered = raw_cropped.copy()
            raw_filtered.filter(l_freq=LOW_FREQ_HZ, h_freq=HIGH_FREQ_HZ, fir_design='firwin',
                              n_jobs=MNE_N_JOBS, verbose=False)
            raw_filtered.notch_filter(freqs=NOTCH_FREQ_HZ, n_jobs=MNE_N_JOBS, verbose=False)

            # Now extract segments from the pre-filtered data
            for segment_start in sequence['segment_starts']:
                # Adjust segment times to be relative to crop start (which includes padding)
                # Example: if segment_start=1950s and crop_tmin=1945s (1950-5 padding),
                # then adjusted_start=5s in the cropped data
                adjusted_start = segment_start - crop_tmin
                adjusted_end = adjusted_start + SEGMENT_DURATION

                # Extract segment from ALREADY FILTERED data (using adjusted times)
                raw_segment = raw_filtered.copy().crop(tmin=adjusted_start, tmax=adjusted_end)

                # Artifact removal (light operation, ok to do per-segment)
                artifact_counts = self.remove_amplitude_artifacts(raw_segment)

                # Get filtered data (DO NOT normalize time-domain signal before STFT!)
                filtered_data = raw_segment.get_data()

                # ✅ CRITICAL FIX: Convert from Volts to microvolts before STFT
                # MNE reads EEG in Volts (~0.0002V), but |STFT|² on such small values
                # produces numerical zeros. Scaling to μV (~200μV) prevents this.
                filtered_data_uv = filtered_data * 1e6

                # STFT - apply to microvolts (NOT volts!)
                stft_coeffs, frequencies, _ = self.apply_stft(filtered_data_uv, sampling_rate)

                # Convert to power spectrograms
                power_spectrogram = np.abs(stft_coeffs) ** 2
                if APPLY_LOG_TRANSFORM:
                    power_spectrogram = np.log10(power_spectrogram + LOG_TRANSFORM_EPSILON)

                # Apply normalization (using training statistics)
                if self.global_mean is not None and self.global_std is not None:
                    power_spectrogram = (power_spectrogram - self.global_mean) / (self.global_std + 1e-8)

                sequence_spectrograms.append(power_spectrogram)

            # Stack into sequence: (seq_len, n_channels, n_freqs, n_times)
            sequence_array = np.stack(sequence_spectrograms, axis=0)

            return {
                'spectrogram': sequence_array,
                'label': 1 if sequence['type'] == self.positive_label else 0,
                'patient_id': sequence['patient_id'],
                'sequence_start_sec': sequence['sequence_start_sec'],
                'sequence_end_sec': sequence['sequence_end_sec'],
                'file_name': sequence['file'],
                'time_to_seizure': sequence.get('time_to_seizure', -1),
                'channel_names': clean_channel_names,
                'frequencies': frequencies
            }

        except Exception as e:
            self.logger.error(f"Failed to process sequence {sequence['patient_id']}/{sequence['file']} at {sequence['sequence_start_sec']}s: {e}")
            self.removed_segments['processing_errors'] += 1
            return None

    def append_to_hdf5(self, split_name: str, batch_sequences: List[Dict]):
        """Append a batch of processed sequences to an HDF5 file.
        Creates the file and datasets if they don't exist.
        """
        if not batch_sequences:
            return

        output_file = self.data_dir / f"{split_name}_dataset.h5"
        self.logger.debug(f"Appending {len(batch_sequences)} sequences to {output_file}")

        # Get dimensions from first sequence
        first_spec = batch_sequences[0]['spectrogram']
        n_seq_len, n_channels, n_freqs, n_times = first_spec.shape

        try:
            with h5py.File(output_file, 'a') as f:
                # Create datasets if they don't exist
                if 'spectrograms' not in f:
                    self.logger.info(f"Creating new HDF5 file or datasets in: {output_file}")
                    # No compression for faster development (can re-enable for production)
                    f.create_dataset('spectrograms',
                                     (0, n_seq_len, n_channels, n_freqs, n_times),
                                     maxshape=(None, n_seq_len, n_channels, n_freqs, n_times),
                                     dtype=np.float32, chunks=True)
                    f.create_dataset('labels', (0,), maxshape=(None,), dtype=np.int32, chunks=True)
                    f.create_dataset('patient_ids', (0,), maxshape=(None,), dtype='S10', chunks=True)
                    
                    seg_info = f.create_group('segment_info')
                    seg_info.create_dataset('start_times', (0,), maxshape=(None,), dtype=np.float32, chunks=True)
                    seg_info.create_dataset('end_times', (0,), maxshape=(None,), dtype=np.float32, chunks=True)
                    seg_info.create_dataset('file_names', (0,), maxshape=(None,), dtype='S20', chunks=True)
                    seg_info.create_dataset('time_to_seizure', (0,), maxshape=(None,), dtype=np.float32, chunks=True)
                    seg_info.create_dataset('segment_ids', (0,), maxshape=(None,), dtype='S50', chunks=True)

                    metadata = f.create_group('metadata')
                    metadata.attrs['task_mode'] = TASK_MODE
                    metadata.attrs['positive_class'] = self.positive_label
                    metadata.attrs['negative_class'] = 'interictal'
                    metadata.attrs['target_channels'] = [ch.encode('utf-8') for ch in batch_sequences[0]['channel_names']]
                    metadata.attrs['frequencies'] = batch_sequences[0]['frequencies']
                    metadata.attrs['frequency_range'] = [LOW_FREQ_HZ, HIGH_FREQ_HZ]
                    metadata.attrs['sequence_length'] = SEQUENCE_LENGTH
                    metadata.attrs['segment_duration'] = SEGMENT_DURATION
                    metadata.attrs['stft_params'] = f"nperseg={STFT_NPERSEG}, noverlap={STFT_NOVERLAP}"
                    metadata.attrs['preprocessing_config'] = json.dumps({
                        'low_freq_hz': LOW_FREQ_HZ,
                        'high_freq_hz': HIGH_FREQ_HZ,
                        'notch_freq_hz': NOTCH_FREQ_HZ,
                        'artifact_threshold_std': ARTIFACT_THRESHOLD_STD,
                        'log_transform': APPLY_LOG_TRANSFORM
                    })
                    metadata.attrs['normalization'] = json.dumps({
                        'method': 'z-score',
                        'global_mean': float(self.global_mean) if self.global_mean is not None else None,
                        'global_std': float(self.global_std) if self.global_std is not None else None,
                        'computed_from': 'training_data'
                    })
                    metadata.attrs['creation_timestamp'] = datetime.now().isoformat()

                # Append data
                n_existing = f['spectrograms'].shape[0]
                n_new = len(batch_sequences)
                new_total = n_existing + n_new

                # Resize datasets
                f['spectrograms'].resize(new_total, axis=0)
                f['labels'].resize(new_total, axis=0)
                f['patient_ids'].resize(new_total, axis=0)
                seg_info = f['segment_info']
                for key in seg_info.keys():
                    seg_info[key].resize(new_total, axis=0)

                # Fill new data
                for i, seq in enumerate(batch_sequences):
                    idx = n_existing + i
                    f['spectrograms'][idx] = seq['spectrogram']
                    f['labels'][idx] = seq['label']
                    f['patient_ids'][idx] = seq['patient_id'].encode('utf-8')
                    seg_info['start_times'][idx] = seq['sequence_start_sec']
                    seg_info['end_times'][idx] = seq['sequence_end_sec']
                    seg_info['file_names'][idx] = seq['file_name'].encode('utf-8')
                    seg_info['time_to_seizure'][idx] = seq['time_to_seizure']
                    sequence_id = f"{seq['patient_id']}_{seq['sequence_start_sec']}"
                    seg_info['segment_ids'][idx] = sequence_id.encode('utf-8')

                # Update metadata
                f['metadata'].attrs['n_sequences'] = new_total
                n_positive = np.sum(f['labels'][:] == 1)
                n_interictal = new_total - n_positive
                f['metadata'].attrs['class_distribution'] = f"{self.positive_label}: {n_positive}, interictal: {n_interictal}"

            self.logger.debug(f"Successfully appended batch. New total sequences for {split_name}: {new_total}")
        except Exception as e:
            self.logger.error(f"Error appending to HDF5 file {output_file}: {e}")
            raise

    def validate_final_dataset(self):
        """Validate the final datasets"""
        self.logger.info("Validating final datasets...")
        
        for split_name in ['train', 'val', 'test']:
            dataset_file = self.data_dir / f"{split_name}_dataset.h5"
            
            if not dataset_file.exists():
                self.logger.error(f"Missing dataset file: {dataset_file}")
                continue
            
            try:
                with h5py.File(dataset_file, 'r') as f:
                    n_segments = f['spectrograms'].shape[0]
                    spec_shape = f['spectrograms'].shape[1:]
                    n_positive = np.sum(f['labels'][:] == 1)
                    n_interictal = np.sum(f['labels'][:] == 0)

                    self.logger.info(f"{split_name} dataset validation:")
                    self.logger.info(f"  - Segments: {n_segments}")
                    self.logger.info(f"  - Spectrogram shape: {spec_shape}")
                    self.logger.info(f"  - Classes: {n_positive} {self.positive_label}, {n_interictal} interictal")
                    self.logger.info(f"  - Balance: {n_positive/n_segments*100:.1f}% {self.positive_label}")
                    
            except Exception as e:
                self.logger.error(f"Error validating {split_name} dataset: {e}")

    def run_preprocessing(self):
        """Main preprocessing pipeline"""
        start_time = time.time()
        
        try:
            # Load segments
            self.logger.info(f"Loading segments from {self.input_json_path}")
            with open(self.input_json_path, 'r') as f:
                data = json.load(f)

            all_sequences = data['sequences']
            self.logger.info(f"Loaded {len(all_sequences)} total sequences")

            # Log validation info from segmentation phase
            if 'validation_info' in data:
                val_info = data['validation_info']
                self.logger.info(f"Channel validation (from segmentation): {val_info.get('files_with_valid_channels', 'N/A')}/{val_info.get('total_files_checked', 'N/A')} files valid")
                if val_info.get('files_with_invalid_channels', 0) > 0:
                    self.logger.info(f"Files skipped due to missing channels: {val_info['files_with_invalid_channels']}")

            # All sequences in the JSON file are already validated during segmentation
            # No need to re-validate here
            valid_sequences = all_sequences
            self.logger.info("Using all sequences (channel validation performed during segmentation)")

            # Determine data splits
            if not self.checkpoint.get('splits_created', False):
                if valid_sequences and all(sequence.get('split') for sequence in valid_sequences):
                    self.logger.info("Detected pre-assigned splits in segmentation metadata; using them directly.")
                    splits = {split_name: [] for split_name in ['train', 'val', 'test']}
                    ignored_sequences = {}

                    for sequence in valid_sequences:
                        split_name = sequence.get('split')
                        if split_name in splits:
                            splits[split_name].append(sequence)
                        else:
                            ignored_sequences[split_name] = ignored_sequences.get(split_name, 0) + 1

                    for split_name, split_sequences in splits.items():
                        positive_count = sum(1 for s in split_sequences if s['type'] == self.positive_label)
                        interictal_count = len(split_sequences) - positive_count
                        self.logger.info(f"{split_name}: {len(split_sequences)} sequences "
                                         f"({positive_count} {self.positive_label}, {interictal_count} interictal)")

                    if any(split_sequences == [] for split_sequences in splits.values()):
                        empty_splits = [name for name, seqs in splits.items() if not seqs]
                        raise ValueError(f"Empty splits detected in segmentation metadata: {empty_splits}")

                    if ignored_sequences:
                        self.logger.warning(f"Ignored {sum(ignored_sequences.values())} sequences with unknown splits: {ignored_sequences}")
                else:
                    self.logger.info("No split metadata found; performing patient-level split.")
                    splits = self.balance_and_split_data(valid_sequences)

                self.checkpoint['splits'] = splits
                self.checkpoint['splits_created'] = True
                self.checkpoint['splits_balanced'] = False  # Force balancing on initial assignment
                self.save_checkpoint()
            else:
                self.logger.info("Using cached data splits")
                splits = self.checkpoint['splits']

            # Ensure splits are balanced (positive vs interictal)
            if not self.checkpoint.get('splits_balanced', False):
                self.logger.info("Balancing splits to enforce equal positive/interictal counts per split...")
                splits, balance_stats = self.balance_split_sequences(splits)

                for split_name in ['train', 'val', 'test']:
                    stats = balance_stats.get(split_name, {})
                    if not stats:
                        continue
                    if stats.get('balanced', False):
                        self.logger.info(
                            f"{split_name}: kept {stats['positive_kept']} {self.positive_label} "
                            f"and {stats['interictal_kept']} interictal "
                            f"(dropped {stats['positive_dropped']} {self.positive_label}, "
                            f"{stats['interictal_dropped']} interictal)"
                        )
                    else:
                        self.logger.warning(
                            f"{split_name}: {stats.get('note', 'Balancing skipped due to missing class')}"
                        )

                self.checkpoint['splits'] = splits
                self.checkpoint['split_balance_stats'] = balance_stats
                self.checkpoint['splits_balanced'] = True
                # Reset processed segments because the sequence list has changed
                self.checkpoint['processed_segments'] = set()

            self.checkpoint['total_sequences'] = sum(len(split_seqs) for split_seqs in splits.values())
            self.save_checkpoint()

            # Compute normalization statistics from training data if not already done
            if not self.checkpoint.get('normalization_stats_computed', False):
                self.logger.info("="*60)
                self.logger.info("COMPUTING NORMALIZATION STATISTICS FROM TRAINING DATA")
                self.logger.info("="*60)

                train_sequences = splits['train']
                train_spectrograms = []

                if not train_sequences:
                    raise ValueError("Training split is empty; cannot compute normalization statistics.")

                self.logger.info(f"Processing {len(train_sequences)} training samples to compute normalization stats...")

                for sequence in tqdm(train_sequences, desc="Computing normalization stats"):
                    processed = self.preprocess_sequence(sequence)
                    if processed:
                        train_spectrograms.append(processed['spectrogram'])

                if train_spectrograms:
                    self.compute_normalization_stats(train_spectrograms)
                    self.checkpoint['normalization_stats_computed'] = True
                    self.save_checkpoint()
                else:
                    raise ValueError("Failed to process any training sequences for normalization stats!")

            # Load normalization statistics before processing
            if not self.load_normalization_stats():
                raise ValueError("Failed to load normalization statistics! Cannot proceed.")

            # Process each split
            for split_name in ['train', 'val', 'test']:
                if split_name in self.checkpoint.get('splits_completed', []):
                    self.logger.info(f"Split {split_name} already completed, skipping")
                    continue
                
                self.logger.info(f"Processing {split_name} split...")
                split_sequences = splits[split_name]

                # Check if HDF5 file already exists and load existing data
                output_file = self.data_dir / f"{split_name}_dataset.h5"
                existing_sequence_ids = set()
                if output_file.exists():
                    try:
                        with h5py.File(output_file, 'r') as f:
                            if 'segment_info' in f and 'segment_ids' in f['segment_info']:
                                existing_ids = f['segment_info']['segment_ids'][:]
                                existing_sequence_ids = {sid.decode('utf-8') for sid in existing_ids}
                        self.logger.info(f"Found existing {split_name} file with {len(existing_sequence_ids)} sequences")
                    except Exception as e:
                        self.logger.warning(f"Could not read existing {split_name} file: {e}")

                batch_sequences = []
                sequences_to_process = []

                # Determine which sequences need processing
                for sequence in split_sequences:
                    sequence_id = f"{sequence['patient_id']}_{sequence['sequence_start_sec']}"
                    if sequence_id not in existing_sequence_ids:
                        sequences_to_process.append(sequence)

                self.logger.info(f"Need to process {len(sequences_to_process)}/{len(split_sequences)} sequences for {split_name}")

                # Process remaining sequences (OPTIMIZED: group by file)
                if sequences_to_process:
                    # Group sequences by file for efficient batch processing
                    file_groups = self.group_sequences_by_file(sequences_to_process)
                    total_files = len(file_groups)

                    self.logger.info(f"Processing {len(sequences_to_process)} sequences from {total_files} unique files")
                    self.logger.info(f"Average {len(sequences_to_process)/total_files:.1f} sequences per file")

                    with tqdm(total=len(sequences_to_process), desc=f"Processing {split_name}") as pbar:
                        # Iterate over file groups (MAJOR OPTIMIZATION: read/filter each file only ONCE)
                        for (patient_id, filename), file_sequences in file_groups.items():
                            # Process ALL sequences from this file in one pass
                            processed_file_sequences = self.preprocess_sequences_from_file(
                                patient_id, filename, file_sequences
                            )

                            # Add successfully processed sequences to batch
                            for sequence, processed_sequence in zip(file_sequences, processed_file_sequences):
                                sequence_id = f"{sequence['patient_id']}_{sequence['sequence_start_sec']}"

                                if processed_sequence:
                                    batch_sequences.append(processed_sequence)
                                    self.checkpoint['processed_segments'].add(sequence_id)
                                    self.checkpoint['processed_count'] += 1

                                pbar.update(1)

                            # Save batch to HDF5 and checkpoint periodically
                            if len(batch_sequences) >= 10:
                                self.append_to_hdf5(split_name, batch_sequences)
                                self.save_checkpoint()
                                batch_sequences = []  # Reset batch

                                # Log progress
                                elapsed = time.time() - start_time
                                rate = self.checkpoint['processed_count'] / elapsed if elapsed > 0 else 0
                                remaining_total = (self.checkpoint['total_sequences'] - self.checkpoint['processed_count'])
                                eta = remaining_total / rate if rate > 0 else 0
                                self.logger.info(f"Progress: {self.checkpoint['processed_count']}/{self.checkpoint['total_sequences']} "
                                               f"({self.checkpoint['processed_count']/self.checkpoint['total_sequences']*100:.1f}%) "
                                               f"- ETA: {eta/60:.1f} minutes")

                    # Save any remaining sequences in the last batch
                    if batch_sequences:
                        self.append_to_hdf5(split_name, batch_sequences)
                        self.save_checkpoint()

                if not sequences_to_process and not existing_sequence_ids:
                    self.logger.warning(f"No sequences processed for {split_name} and no existing file found")
                
                # Log removal statistics for this split
                total_removed = sum(self.removed_segments.values())
                if total_removed > 0:
                    self.logger.info(f"{split_name} split - Removed segments: {total_removed} total "
                                   f"({self.removed_segments['wrong_duration']} wrong duration, "
                                   f"{self.removed_segments['beyond_file_bounds']} beyond bounds, "
                                   f"{self.removed_segments['processing_errors']} processing errors)")
                    # Reset counters for next split
                    self.removed_segments = {'wrong_duration': 0, 'beyond_file_bounds': 0, 'processing_errors': 0}
                
                # Mark split as completed
                if 'splits_completed' not in self.checkpoint:
                    self.checkpoint['splits_completed'] = []
                self.checkpoint['splits_completed'].append(split_name)
                self.save_checkpoint()
            
            # Final validation
            self.validate_final_dataset()
            
            # Final statistics
            total_time = time.time() - start_time
            self.logger.info("="*60)
            self.logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total time: {total_time/60:.1f} minutes")
            self.logger.info(f"Processed segments: {self.checkpoint['processed_count']}")
            self.logger.info(f"Average rate: {self.checkpoint['processed_count']/total_time:.1f} segments/second")
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise

if __name__ == "__main__":
    # Run preprocessing
    preprocessor = EEGPreprocessor()
    preprocessor.run_preprocessing()
