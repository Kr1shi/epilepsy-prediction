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
        
    def validate_segments(self, segments: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Validate segments have all required channels"""
        self.logger.info("Validating segment channels...")
        
        valid_segments = []
        validation_stats = {
            'total_segments': len(segments),
            'valid_segments': 0,
            'invalid_segments': 0,
            'patients_processed': set(),
            'patients_with_no_valid': set(),
            'missing_files': 0
        }
        
        for segment in tqdm(segments, desc="Validating channels"):
            try:
                # Construct EDF path
                edf_path = f"physionet.org/files/chbmit/1.0.0/{segment['patient_id']}/{segment['file']}"
                
                if not os.path.exists(edf_path):
                    self.logger.warning(f"File not found: {edf_path}")
                    validation_stats['missing_files'] += 1
                    continue
                
                # Read EDF header only (fast)
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                available_channels = set(raw.ch_names)
                
                # Check for all target channels (including duplicates)
                missing_channels = []
                for target_ch in TARGET_CHANNELS:
                    if target_ch not in available_channels:
                        # Check for renamed duplicates
                        duplicate_matches = [ch for ch in available_channels 
                                           if ch.startswith(target_ch + '-') and ch.split('-')[-1].isdigit()]
                        if not duplicate_matches:
                            missing_channels.append(target_ch)
                
                validation_stats['patients_processed'].add(segment['patient_id'])
                
                if missing_channels:
                    self.logger.debug(f"Segment {segment['patient_id']}/{segment['file']} missing channels: {missing_channels}")
                    validation_stats['invalid_segments'] += 1
                else:
                    valid_segments.append(segment)
                    validation_stats['valid_segments'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error validating segment {segment.get('patient_id', 'unknown')}/{segment.get('file', 'unknown')}: {e}")
                validation_stats['invalid_segments'] += 1
        
        # Check for patients with no valid segments
        patient_valid_counts = {}
        for segment in valid_segments:
            patient_id = segment['patient_id']
            patient_valid_counts[patient_id] = patient_valid_counts.get(patient_id, 0) + 1
        
        for patient_id in validation_stats['patients_processed']:
            if patient_id not in patient_valid_counts:
                validation_stats['patients_with_no_valid'].add(patient_id)
                self.logger.warning(f"Patient {patient_id} has 0 valid segments")
        
        # Convert sets to lists for JSON serialization
        validation_stats['patients_processed'] = list(validation_stats['patients_processed'])
        validation_stats['patients_with_no_valid'] = list(validation_stats['patients_with_no_valid'])
        
        self.logger.info(f"Validation complete: {validation_stats['valid_segments']}/{validation_stats['total_segments']} segments valid")
        self.logger.info(f"Patients with valid segments: {len(patient_valid_counts)}")
        
        return valid_segments, validation_stats

    def balance_and_split_data(self, segments: List[Dict]) -> Dict[str, List[Dict]]:
        """Balance classes globally and split into train/val/test"""
        self.logger.info("Balancing and splitting data...")
        
        # Separate by class
        preictal_segments = [s for s in segments if s['type'] == 'preictal']
        interictal_segments = [s for s in segments if s['type'] == 'interictal']
        
        self.logger.info(f"Before balancing: {len(preictal_segments)} preictal, {len(interictal_segments)} interictal")
        
        # Balance to minimum class
        min_count = min(len(preictal_segments), len(interictal_segments))
        
        # Randomly sample to balance (using RandomState for reproducibility)
        rng = np.random.RandomState(42)
        balanced_preictal = rng.choice(preictal_segments, size=min_count, replace=False).tolist()
        balanced_interictal = rng.choice(interictal_segments, size=min_count, replace=False).tolist()
        
        # Combine and shuffle
        balanced_segments = balanced_preictal + balanced_interictal
        rng.shuffle(balanced_segments)
        
        self.logger.info(f"After balancing: {len(balanced_preictal)} preictal, {len(balanced_interictal)} interictal")
        self.logger.info(f"Total balanced segments: {len(balanced_segments)}")
        
        # Split into train/val/test
        train_segments, temp_segments = train_test_split(
            balanced_segments, 
            test_size=(self.val_ratio + self.test_ratio),
            stratify=[s['type'] for s in balanced_segments],
            random_state=42
        )
        
        val_segments, test_segments = train_test_split(
            temp_segments,
            test_size=self.test_ratio / (self.val_ratio + self.test_ratio),
            stratify=[s['type'] for s in temp_segments],
            random_state=42
        )
        
        splits = {
            'train': train_segments,
            'val': val_segments, 
            'test': test_segments
        }
        
        # Log split statistics
        for split_name, split_segments in splits.items():
            preictal_count = sum(1 for s in split_segments if s['type'] == 'preictal')
            interictal_count = len(split_segments) - preictal_count
            self.logger.info(f"{split_name}: {len(split_segments)} segments ({preictal_count} preictal, {interictal_count} interictal)")
        
        return splits

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
        
        # Check for flat channels
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
            
            # Filter to desired frequency range
            freq_mask = (f >= LOW_FREQ_HZ) & (f <= HIGH_FREQ_HZ)
            filtered_spec = Zxx[freq_mask, :]
            spectrograms.append(filtered_spec)
        
        stft_spectrograms = np.array(spectrograms)
        frequencies = f[freq_mask]
        
        return stft_spectrograms, frequencies

    def preprocess_segment(self, segment: Dict) -> Optional[Dict]:
        """Preprocess a single segment"""
        try:
            # Construct EDF path
            edf_path = f"physionet.org/files/chbmit/1.0.0/{segment['patient_id']}/{segment['file']}"
            
            # Read EDF file
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            sampling_rate = raw.info['sfreq']
            file_duration = raw.times[-1]  # Get actual file duration
            
            # Check if segment is exactly SEGMENT_DURATION seconds and within file bounds
            expected_duration = segment['end_sec'] - segment['start_sec']
            if expected_duration != SEGMENT_DURATION:
                self.logger.debug(f"Segment duration not {SEGMENT_DURATION}s ({expected_duration}s), skipping: {segment['patient_id']}/{segment['file']} at {segment['start_sec']}s")
                self.removed_segments['wrong_duration'] += 1
                return None
                
            if segment['end_sec'] > file_duration:
                self.logger.debug(f"Segment extends beyond file duration ({segment['end_sec']}s > {file_duration:.1f}s), skipping: {segment['patient_id']}/{segment['file']} at {segment['start_sec']}s")
                self.removed_segments['beyond_file_bounds'] += 1
                return None
            
            # Select target channels
            raw_selected, clean_channel_names = self.select_target_channels(raw)
            
            # Extract segment (we know it's valid SEGMENT_DURATION seconds now)
            raw_segment = raw_selected.copy().crop(tmin=segment['start_sec'], tmax=segment['end_sec'])
            
            # Bad channel detection and interpolation
            bad_channels = self.detect_bad_channels(raw_segment)
            if bad_channels:
                raw_segment.info['bads'] = bad_channels
                # Only interpolate if digitization info is available
                if raw_segment.info.get('dig') is not None:
                    raw_segment.interpolate_bads(reset_bads=True, verbose=False)
                    self.logger.debug(f"Interpolated bad channels: {bad_channels}")
                else:
                    self.logger.debug(f"Skipping interpolation for bad channels {bad_channels} - no digitization info")
                    # Reset bad channels list since we can't interpolate
                    raw_segment.info['bads'] = []
            
            # Artifact removal
            artifact_counts = self.remove_amplitude_artifacts(raw_segment)
            
            # Filtering
            raw_filtered = raw_segment.copy()
            raw_filtered.filter(l_freq=LOW_FREQ_HZ, h_freq=HIGH_FREQ_HZ, fir_design='firwin', verbose=False)
            raw_filtered.notch_filter(freqs=NOTCH_FREQ_HZ, verbose=False)
            
            # Normalization
            filtered_data = raw_filtered.get_data()
            normalized_data = self.robust_normalize(filtered_data)
            
            # STFT
            stft_coeffs, frequencies = self.apply_stft(normalized_data, sampling_rate)
            
            # Convert to power spectrograms
            power_spectrograms = np.abs(stft_coeffs) ** 2
            if APPLY_LOG_TRANSFORM:
                power_spectrograms = np.log10(power_spectrograms + LOG_TRANSFORM_EPSILON)
            
            # Create segment identifier
            segment_id = f"{segment['patient_id']}_{segment['type']}_{segment['start_sec']}"
            
            return {
                'segment_id': segment_id,
                'spectrogram': power_spectrograms.astype(np.float32),
                'label': 1 if segment['type'] == 'preictal' else 0,
                'patient_id': segment['patient_id'],
                'start_sec': segment['start_sec'],
                'end_sec': segment['end_sec'],  # Always SEGMENT_DURATION seconds now
                'file_name': segment['file'],
                'time_to_seizure': segment.get('time_to_seizure', -1),
                'channel_names': clean_channel_names,
                'frequencies': frequencies,
                'sampling_rate': sampling_rate,
                'bad_channels': bad_channels,
                'artifact_count': np.sum(artifact_counts)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process segment {segment['patient_id']}/{segment['file']} at {segment['start_sec']}s: {e}")
            self.removed_segments['processing_errors'] += 1
            return None

    def append_to_hdf5(self, split_name: str, batch_segments: List[Dict]):
        """Append a batch of processed segments to an HDF5 file.
        Creates the file and datasets if they don't exist.
        """
        if not batch_segments:
            return

        output_file = self.data_dir / f"{split_name}_dataset.h5"
        self.logger.debug(f"Appending {len(batch_segments)} segments to {output_file}")

        # Get dimensions from first segment
        first_spec = batch_segments[0]['spectrogram']
        n_channels, n_freqs, n_times = first_spec.shape
        
        try:
            with h5py.File(output_file, 'a') as f:
                # Create datasets if they don't exist
                if 'spectrograms' not in f:
                    self.logger.info(f"Creating new HDF5 file or datasets in: {output_file}")
                    f.create_dataset('spectrograms', 
                                     (0, n_channels, n_freqs, n_times), 
                                     maxshape=(None, n_channels, n_freqs, n_times),
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
                    metadata.attrs['target_channels'] = [ch.encode('utf-8') for ch in batch_segments[0]['channel_names']]
                    metadata.attrs['frequencies'] = batch_segments[0]['frequencies']
                    metadata.attrs['frequency_range'] = [LOW_FREQ_HZ, HIGH_FREQ_HZ]
                    metadata.attrs['sampling_rate'] = batch_segments[0]['sampling_rate']
                    metadata.attrs['stft_params'] = f"nperseg={STFT_NPERSEG}, noverlap={STFT_NOVERLAP}"
                    metadata.attrs['preprocessing_config'] = json.dumps({
                        'low_freq_hz': LOW_FREQ_HZ,
                        'high_freq_hz': HIGH_FREQ_HZ,
                        'notch_freq_hz': NOTCH_FREQ_HZ,
                        'artifact_threshold_std': ARTIFACT_THRESHOLD_STD,
                        'log_transform': APPLY_LOG_TRANSFORM
                    })
                    metadata.attrs['creation_timestamp'] = datetime.now().isoformat()

                # Append data
                n_existing = f['spectrograms'].shape[0]
                n_new = len(batch_segments)
                new_total = n_existing + n_new

                # Resize datasets
                f['spectrograms'].resize(new_total, axis=0)
                f['labels'].resize(new_total, axis=0)
                f['patient_ids'].resize(new_total, axis=0)
                seg_info = f['segment_info']
                for key in seg_info.keys():
                    seg_info[key].resize(new_total, axis=0)

                # Fill new data
                for i, seg in enumerate(batch_segments):
                    idx = n_existing + i
                    f['spectrograms'][idx] = seg['spectrogram']
                    f['labels'][idx] = seg['label']
                    f['patient_ids'][idx] = seg['patient_id'].encode('utf-8')
                    seg_info['start_times'][idx] = seg['start_sec']
                    seg_info['end_times'][idx] = seg['end_sec']
                    seg_info['file_names'][idx] = seg['file_name'].encode('utf-8')
                    seg_info['time_to_seizure'][idx] = seg['time_to_seizure']
                    seg_info['segment_ids'][idx] = seg['segment_id'].encode('utf-8')

                # Update metadata
                f['metadata'].attrs['n_segments'] = new_total
                n_preictal = np.sum(f['labels'][:] == 1)
                n_interictal = new_total - n_preictal
                f['metadata'].attrs['class_distribution'] = f"preictal: {n_preictal}, interictal: {n_interictal}"

            self.logger.debug(f"Successfully appended batch. New total segments for {split_name}: {new_total}")
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
                    n_preictal = np.sum(f['labels'][:] == 1)
                    n_interictal = np.sum(f['labels'][:] == 0)
                    
                    self.logger.info(f"{split_name} dataset validation:")
                    self.logger.info(f"  - Segments: {n_segments}")
                    self.logger.info(f"  - Spectrogram shape: {spec_shape}")
                    self.logger.info(f"  - Classes: {n_preictal} preictal, {n_interictal} interictal")
                    self.logger.info(f"  - Balance: {n_preictal/n_segments*100:.1f}% preictal")
                    
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
            
            all_segments = data['preictal_segments'] + data['interictal_segments']
            self.logger.info(f"Loaded {len(all_segments)} total segments")
            
            # Check if we need to validate segments
            if SKIP_CHANNEL_VALIDATION:
                if ('valid_segments' in self.checkpoint and 
                    'segments' in self.checkpoint.get('valid_segments', {}) and
                    self.checkpoint['valid_segments'].get('segments')):
                    
                    self.logger.info("Skipping channel validation (SKIP_CHANNEL_VALIDATION=True) - using cache")
                    valid_segments = self.checkpoint['valid_segments']['segments']
                    self.logger.info(f"Using cached {len(valid_segments)} valid segments")
                else:
                    self.logger.info("Skipping channel validation (SKIP_CHANNEL_VALIDATION=True) - using all segments")
                    valid_segments = all_segments
                    # Cache the decision to skip validation
                    self.checkpoint['valid_segments'] = {
                        'segments': valid_segments,
                        'stats': {
                            'total_segments': len(all_segments),
                            'valid_segments': len(all_segments),
                            'invalid_segments': 0,
                            'validation_skipped': True
                        }
                    }
                    self.save_checkpoint()
                    
            elif ('valid_segments' not in self.checkpoint or 
                'segments' not in self.checkpoint.get('valid_segments', {}) or
                not self.checkpoint['valid_segments'].get('segments')):
                
                self.logger.info("Performing channel validation...")
                # Validate segments
                valid_segments, validation_stats = self.validate_segments(all_segments)
                self.checkpoint['valid_segments'] = {
                    'segments': valid_segments,
                    'stats': validation_stats
                }
                self.save_checkpoint()
            else:
                self.logger.info("Using cached segment validation")
                valid_segments = self.checkpoint['valid_segments']['segments']
                self.logger.info(f"Loaded {len(valid_segments)} valid segments from cache")
            
            # Balance and split data
            if not self.checkpoint.get('splits_created', False):
                splits = self.balance_and_split_data(valid_segments)
                self.checkpoint['splits'] = splits
                self.checkpoint['splits_created'] = True
                self.checkpoint['total_segments'] = sum(len(split_segments) for split_segments in splits.values())
                self.save_checkpoint()
            else:
                self.logger.info("Using cached data splits")
                splits = self.checkpoint['splits']
            
            # Process each split
            for split_name in ['train', 'val', 'test']:
                if split_name in self.checkpoint.get('splits_completed', []):
                    self.logger.info(f"Split {split_name} already completed, skipping")
                    continue
                
                self.logger.info(f"Processing {split_name} split...")
                split_segments = splits[split_name]
                
                # Check if HDF5 file already exists and load existing data
                output_file = self.data_dir / f"{split_name}_dataset.h5"
                existing_segment_ids = set()
                if output_file.exists():
                    try:
                        with h5py.File(output_file, 'r') as f:
                            if 'segment_info' in f and 'segment_ids' in f['segment_info']:
                                existing_ids = f['segment_info']['segment_ids'][:]
                                existing_segment_ids = {sid.decode('utf-8') for sid in existing_ids}
                        self.logger.info(f"Found existing {split_name} file with {len(existing_segment_ids)} segments")
                    except Exception as e:
                        self.logger.warning(f"Could not read existing {split_name} file: {e}")
                
                batch_segments = []
                segments_to_process = []

                # Determine which segments need processing
                for segment in split_segments:
                    segment_id = f"{segment['patient_id']}_{segment['type']}_{segment['start_sec']}"
                    if segment_id not in existing_segment_ids:
                        segments_to_process.append(segment)

                self.logger.info(f"Need to process {len(segments_to_process)}/{len(split_segments)} segments for {split_name}")

                # Process remaining segments
                if segments_to_process:
                    with tqdm(total=len(segments_to_process), desc=f"Processing {split_name}") as pbar:
                        for segment in segments_to_process:
                            segment_id = f"{segment['patient_id']}_{segment['type']}_{segment['start_sec']}"
                            
                            # Process segment
                            processed_segment = self.preprocess_segment(segment)
                            
                            if processed_segment:
                                batch_segments.append(processed_segment)
                                self.checkpoint['processed_segments'].add(segment_id)
                                self.checkpoint['processed_count'] += 1
                            
                            pbar.update(1)
                            
                            # Save batch to HDF5 and checkpoint periodically
                            if len(batch_segments) >= self.checkpoint_interval:
                                self.append_to_hdf5(split_name, batch_segments)
                                self.save_checkpoint()
                                batch_segments = [] # Reset batch
                                
                                # Log progress
                                elapsed = time.time() - start_time
                                rate = self.checkpoint['processed_count'] / elapsed if elapsed > 0 else 0
                                remaining_total = (self.checkpoint['total_segments'] - self.checkpoint['processed_count'])
                                eta = remaining_total / rate if rate > 0 else 0
                                self.logger.info(f"Progress: {self.checkpoint['processed_count']}/{self.checkpoint['total_segments']} "
                                               f"({self.checkpoint['processed_count']/self.checkpoint['total_segments']*100:.1f}%) "
                                               f"- ETA: {eta/60:.1f} minutes")

                    # Save any remaining segments in the last batch
                    if batch_segments:
                        self.append_to_hdf5(split_name, batch_segments)
                        self.save_checkpoint()

                if not segments_to_process and not existing_segment_ids:
                     self.logger.warning(f"No segments processed for {split_name} and no existing file found")
                
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