import mne
import numpy as np
import h5py
import json
import os
import logging
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from scipy.signal import spectrogram
from data_segmentation_helpers.config import *
import matplotlib.pyplot as plt
import random

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
    def __init__(self, input_json_path: str = 'all_patients_segments.json'):
        self.input_json_path = input_json_path
        self.output_dir = Path("preprocessing")
        self.data_dir = self.output_dir / "data"
        self.logs_dir = self.output_dir / "logs"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_file = self.checkpoint_dir / "progress.json"
        
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
        
        # Initialize logging and directories
        self.setup_logging_and_directories()
        
        # Load checkpoint if exists
        self.checkpoint = self.load_checkpoint()
        
        self.logger.info("EEG Preprocessor initialized")
        self.logger.info(f"Target channels: {TARGET_CHANNELS}")
        self.logger.info(f"Filter settings: {LOW_FREQ_HZ}-{HIGH_FREQ_HZ} Hz, notch: {NOTCH_FREQ_HZ} Hz")
        self.logger.info(f"Segment duration: {SEGMENT_DURATION} seconds")
        self.logger.info(f"Skip channel validation: {SKIP_CHANNEL_VALIDATION}")
    
    def setup_logging_and_directories(self):
        """Setup logging and create directory structure"""
        # Create directories
        for dir_path in [self.data_dir, self.logs_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.logs_dir / "visualization.log"
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
        self.logger.info("EEG VISUALIZATION SCRIPT STARTED")
        self.logger.info("="*60)

    def load_checkpoint(self) -> Dict:
        """Load checkpoint if exists, handle corrupted files"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                
                if isinstance(checkpoint.get('processed_segments'), list):
                    checkpoint['processed_segments'] = set(checkpoint['processed_segments'])
                
                self.logger.info(f"Loaded checkpoint: {len(checkpoint.get('processed_segments', []))} segments processed")
                return checkpoint
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                self.logger.warning(f"Corrupted checkpoint file detected: {e}")
                self.logger.warning("Starting fresh - deleting corrupted checkpoint")
                backup_path = self.checkpoint_file.with_suffix('.json.corrupted')
                self.checkpoint_file.rename(backup_path)
                self.logger.info(f"Corrupted checkpoint backed up to: {backup_path}")
        
        return {
            "processed_segments": set(),
            "current_split": None,
            "splits_completed": [],
            "total_segments": 0,
            "processed_count": 0,
            "start_time": datetime.now().isoformat(),
            "valid_segments": {}
        }

    def save_checkpoint(self):
        """Save current progress, handling non-JSON serializable objects"""
        checkpoint_copy = self.checkpoint.copy()
        
        if isinstance(checkpoint_copy.get('processed_segments'), set):
            checkpoint_copy['processed_segments'] = list(checkpoint_copy['processed_segments'])
        
        if 'valid_segments' in checkpoint_copy and 'stats' in checkpoint_copy['valid_segments']:
            stats = checkpoint_copy['valid_segments']['stats']
            if 'patients_processed' in stats and isinstance(stats['patients_processed'], set):
                stats['patients_processed'] = list(stats['patients_processed'])
            if 'patients_with_no_valid' in stats and isinstance(stats['patients_with_no_valid'], set):
                stats['patients_with_no_valid'] = list(stats['patients_with_no_valid'])
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_copy, f, indent=2)
        
    def select_target_channels(self, raw, target_channels=TARGET_CHANNELS):
        """Select target channels, handling duplicates"""
        available_channels = []
        clean_channel_names = []
        
        for ch in target_channels:
            if ch in raw.ch_names:
                available_channels.append(ch)
                clean_channel_names.append(ch)
            else:
                duplicate_matches = [name for name in raw.ch_names 
                                   if name.startswith(ch + '-') and name.split('-')[-1].isdigit()]
                if duplicate_matches:
                    available_channels.append(duplicate_matches[0])
                    clean_channel_names.append(ch)
        
        raw_filtered = raw.copy().pick(available_channels)
        return raw_filtered, clean_channel_names

    def detect_bad_channels(self, raw):
        """Detect bad channels using statistical methods"""
        data = raw.get_data()
        bad_channels = []
        
        channel_stds = np.std(data, axis=1)
        median_std = np.median(channel_stds)
        mad_std = np.median(np.abs(channel_stds - median_std))
        
        for i, std_val in enumerate(channel_stds):
            z_score = np.abs((std_val - median_std) / (mad_std * 1.4826))
            if z_score > BAD_CHANNEL_STD_THRESHOLD:
                bad_channels.append(raw.ch_names[i])
        
        flat_threshold = np.percentile(channel_stds, BAD_CHANNEL_FLAT_PERCENTILE)
        for i, std_val in enumerate(channel_stds):
            if std_val < flat_threshold * 0.1:
                if raw.ch_names[i] not in bad_channels:
                    bad_channels.append(raw.ch_names[i])
        
        return bad_channels

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
        """Apply STFT to create spectrograms"""
        spectrograms = []
        
        for ch in range(data.shape[0]):
            f, t, Zxx = spectrogram(
                data[ch, :], 
                fs=sampling_rate, 
                nperseg=STFT_NPERSEG, 
                noverlap=STFT_NOVERLAP,
                window='hann'
            )
            
            freq_mask = (f >= LOW_FREQ_HZ) & (f <= HIGH_FREQ_HZ)
            filtered_spec = Zxx[freq_mask, :]
            spectrograms.append(filtered_spec)
        
        stft_spectrograms = np.array(spectrograms)
        frequencies = f[freq_mask]
        
        return stft_spectrograms, frequencies

    def preprocess_segment(self, segment: Dict) -> Optional[Dict]:
        """Preprocess a single segment"""
        try:
            edf_path = f"physionet.org/files/chbmit/1.0.0/{segment['patient_id']}/{segment['file']}"
            
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            sampling_rate = raw.info['sfreq']
            file_duration = raw.times[-1]
            
            expected_duration = segment['end_sec'] - segment['start_sec']
            if expected_duration != SEGMENT_DURATION:
                self.logger.debug(f"Segment duration not {SEGMENT_DURATION}s, skipping.")
                return None
                
            if segment['end_sec'] > file_duration:
                self.logger.debug(f"Segment extends beyond file duration, skipping.")
                return None
            
            raw_selected, clean_channel_names = self.select_target_channels(raw)
            raw_segment = raw_selected.copy().crop(tmin=segment['start_sec'], tmax=segment['end_sec'])
            
            bad_channels = self.detect_bad_channels(raw_segment)
            if bad_channels:
                raw_segment.info['bads'] = bad_channels
                if raw_segment.info.get('dig') is not None:
                    raw_segment.interpolate_bads(reset_bads=True, verbose=False)
                else:
                    raw_segment.info['bads'] = []
            
            artifact_counts = self.remove_amplitude_artifacts(raw_segment)
            
            raw_filtered = raw_segment.copy()
            raw_filtered.filter(l_freq=LOW_FREQ_HZ, h_freq=HIGH_FREQ_HZ, fir_design='firwin', verbose=False)
            raw_filtered.notch_filter(freqs=NOTCH_FREQ_HZ, verbose=False)
            
            filtered_data = raw_filtered.get_data()
            normalized_data = self.robust_normalize(filtered_data)
            
            stft_coeffs, frequencies = self.apply_stft(normalized_data, sampling_rate)
            
            power_spectrograms = np.abs(stft_coeffs) ** 2
            if APPLY_LOG_TRANSFORM:
                power_spectrograms = np.log10(power_spectrograms + LOG_TRANSFORM_EPSILON)
            
            segment_id = f"{segment['patient_id']}_{segment['type']}_{segment['start_sec']}"
            
            return {
                'segment_id': segment_id,
                'spectrogram': power_spectrograms.astype(np.float32),
                'label': 1 if segment['type'] == 'preictal' else 0,
                'patient_id': segment['patient_id'],
                'start_sec': segment['start_sec'],
                'end_sec': segment['end_sec'],
                'file_name': segment['file'],
                'time_to_seizure': segment.get('time_to_seizure', -1),
                'channel_names': clean_channel_names,
                'frequencies': frequencies,
                'sampling_rate': sampling_rate,
                'bad_channels': bad_channels,
                'artifact_count': np.sum(artifact_counts)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process segment {segment['patient_id']}/{segment['file']}: {e}")
            return None

class EEGVisualizer(EEGPreprocessor):
    def __init__(self, input_json_path: str = 'all_patients_segments.json'):
        super().__init__(input_json_path)
        self.visualization_dir = self.output_dir / "visualizations"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Visualizations will be saved to: {self.visualization_dir}")

    def plot_spectrogram(self, processed_segment: Dict):
        """Plots and saves the spectrogram for a single processed segment."""
        if not processed_segment:
            self.logger.warning("Cannot plot an empty segment.")
            return

        spec_data = processed_segment['spectrogram']
        patient_id = processed_segment['patient_id']
        segment_type = "preictal" if processed_segment['label'] == 1 else "interictal"
        start_sec = processed_segment['start_sec']
        channel_names = processed_segment['channel_names']
        frequencies = processed_segment['frequencies']
        
        n_channels = spec_data.shape[0]
        
        fig, axes = plt.subplots(n_channels, 1, figsize=(12, n_channels * 2.5), sharex=True, sharey=True)
        if n_channels == 1:
            axes = [axes]

        fig.suptitle(f'Spectrogram for {patient_id} - {segment_type} segment @ {start_sec}s', fontsize=16)

        time_bins = spec_data.shape[2]
        extent = [0, SEGMENT_DURATION, frequencies[0], frequencies[-1]]

        for i in range(n_channels):
            ax = axes[i]
            im = ax.imshow(spec_data[i, :, :], aspect='auto', origin='lower', extent=extent, cmap='viridis')
            ax.set_ylabel(f"{channel_names[i]}\nFreq (Hz)")
            
        axes[-1].set_xlabel("Time (s)")
        
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, label='Log Power')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        output_filename = self.visualization_dir / f"{patient_id}_{segment_type}_{start_sec}s.png"
        plt.savefig(output_filename)
        self.logger.info(f"Saved plot to {output_filename}")
        plt.close(fig)

    def run_visualization(self, num_examples: int = 2):
        """Process and visualize a few random example segments."""
        self.logger.info(f"Starting visualization process for {num_examples} preictal and {num_examples} interictal examples.")
        
        try:
            self.logger.info(f"Loading segments from {self.input_json_path}")
            with open(self.input_json_path, 'r') as f:
                data = json.load(f)
            
            preictal_segments = data['preictal_segments']
            interictal_segments = data['interictal_segments']
            
            random_preictal = random.sample(preictal_segments, min(num_examples, len(preictal_segments)))
            random_interictal = random.sample(interictal_segments, min(num_examples, len(interictal_segments)))
            
            segments_to_visualize = random_preictal + random_interictal
            
            self.logger.info(f"Selected {len(segments_to_visualize)} segments to visualize.")

            for segment in tqdm(segments_to_visualize, desc="Processing and Visualizing Segments"):
                self.logger.info(f"Processing segment: {segment['patient_id']}/{segment['file']} at {segment['start_sec']}s")
                processed_data = self.preprocess_segment(segment)
                
                if processed_data:
                    self.plot_spectrogram(processed_data)
                else:
                    self.logger.warning(f"Skipping visualization for segment due to processing failure.")

            self.logger.info("="*60)
            self.logger.info("VISUALIZATION COMPLETED SUCCESSFULLY")
            self.logger.info("="*60)

        except Exception as e:
            self.logger.error(f"Visualization process failed: {e}")
            raise

if __name__ == "__main__":
    visualizer = EEGVisualizer()
    visualizer.run_visualization(num_examples=3)
