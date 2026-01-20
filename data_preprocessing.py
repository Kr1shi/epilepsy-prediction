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
from scipy.signal import spectrogram
from multiprocessing import cpu_count
from data_segmentation_helpers.config import (
    TASK_MODE,
    SEGMENT_DURATION,
    SEQUENCE_LENGTH,
    BASE_PATH,
    TARGET_CHANNELS,
    LOW_FREQ_HZ,
    HIGH_FREQ_HZ,
    NOTCH_FREQ_HZ,
    STFT_NPERSEG,
    STFT_NOVERLAP,
    ARTIFACT_THRESHOLD_STD,
    LOG_TRANSFORM_EPSILON,
    APPLY_LOG_TRANSFORM,
    PREPROCESSING_WORKERS,
    MNE_N_JOBS,
    LOPO_FOLD_ID,
    LOPO_PATIENTS,
    get_fold_config,
)

# Suppress MNE warnings that don't affect processing
warnings.filterwarnings("ignore", message="Channel names are not unique")
warnings.filterwarnings("ignore", message=".*duplicates.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Scaling factor is not defined")
warnings.filterwarnings("ignore", message=".*scaling factor.*", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore", message="Number of records from the header does not match"
)
warnings.filterwarnings("ignore", message=".*file size.*", category=RuntimeWarning)

# Set MNE log level early
mne.set_log_level("ERROR")


class EEGPreprocessor:
    def __init__(self, fold_config: Dict):
        """Initialize preprocessor with fold-specific configuration.

        Args:
            fold_config: Dictionary from get_fold_config() with output_prefix, random_seed, etc.
        """
        output_prefix = fold_config["output_prefix"]
        balance_seed = fold_config["random_seed"]

        # Input JSON from segmentation
        self.input_json_path = f"{output_prefix}_sequences_{TASK_MODE}.json"

        # Output directories
        self.output_dir = Path("preprocessing")
        self.dataset_prefix = output_prefix
        self.data_dir = self.output_dir / "data" / self.dataset_prefix
        self.logs_dir = self.output_dir / "logs" / self.dataset_prefix
        self.checkpoint_dir = self.output_dir / "checkpoints" / self.dataset_prefix
        self.checkpoint_file = self.checkpoint_dir / "progress.json"
        self.balance_seed = balance_seed

        # Processing settings
        self.checkpoint_interval = 50

        # Statistics tracking
        self.removed_segments = {
            "wrong_duration": 0,
            "beyond_file_bounds": 0,
            "processing_errors": 0,
        }

        # Initialize logging and directories
        self.setup_logging_and_directories()

        # Load checkpoint if exists
        self.checkpoint = self.load_checkpoint()

        self.logger.info(
            "EEG Preprocessor initialized (Per-Sequence Zero-Centering Mode)"
        )
        self.logger.info(f"Target channels: {TARGET_CHANNELS}")
        self.logger.info(
            f"Filter settings: {LOW_FREQ_HZ}-{HIGH_FREQ_HZ} Hz, notch: {NOTCH_FREQ_HZ} Hz"
        )
        self.logger.info(f"Segment duration: {SEGMENT_DURATION} seconds")

    @property
    def positive_label(self):
        """Get positive class label based on task mode"""
        return "preictal" if TASK_MODE == "prediction" else "ictal"

    @staticmethod
    def group_sequences_by_file(
        sequences: List[Dict],
    ) -> Dict[Tuple[str, str], List[Dict]]:
        """Group sequences by (patient_id, filename) for efficient batch processing."""
        from collections import defaultdict

        file_groups = defaultdict(list)
        for sequence in sequences:
            key = (sequence["patient_id"], sequence["file"])
            file_groups[key].append(sequence)
        return dict(file_groups)

    def setup_logging_and_directories(self):
        """Setup logging and create directory structure"""
        for dir_path in [self.data_dir, self.logs_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        log_file = self.logs_dir / "preprocessing.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file, mode="a"), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 60)
        self.logger.info("EEG PREPROCESSING PIPELINE STARTED")
        self.logger.info("=" * 60)

    def load_checkpoint(self) -> Dict:
        """Load checkpoint if exists"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    checkpoint = json.load(f)

                # OPTIMIZATION: Use completed_files instead of processed_segments
                if "completed_files" in checkpoint:
                    if isinstance(checkpoint["completed_files"], list):
                        checkpoint["completed_files"] = set(
                            checkpoint["completed_files"]
                        )
                else:
                    checkpoint["completed_files"] = set()

                # Cleanup legacy processed_segments to reduce memory/disk overhead
                if "processed_segments" in checkpoint:
                    del checkpoint["processed_segments"]

                return checkpoint
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint: {e}")

        return {
            "completed_files": set(),
            "processed_count": 0,
            "start_time": datetime.now().isoformat(),
        }

    def save_checkpoint(self):
        """Save current progress"""
        checkpoint_copy = self.checkpoint.copy()

        # Serialize completed_files set to list
        if isinstance(checkpoint_copy.get("completed_files"), set):
            checkpoint_copy["completed_files"] = list(
                checkpoint_copy["completed_files"]
            )

        # Ensure legacy key is gone
        if "processed_segments" in checkpoint_copy:
            del checkpoint_copy["processed_segments"]

        with open(self.checkpoint_file, "w") as f:
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
                duplicate_matches = [
                    name
                    for name in raw.ch_names
                    if name.startswith(ch + "-") and name.split("-")[-1].isdigit()
                ]
                if duplicate_matches:
                    available_channels.append(duplicate_matches[0])
                    clean_channel_names.append(ch)

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

        for ch in range(data.shape[0]):
            if np.any(artifact_mask[ch, :]):
                artifact_indices = np.where(artifact_mask[ch, :])[0]
                clean_indices = np.where(~artifact_mask[ch, :])[0]
                if len(clean_indices) > 10:
                    data[ch, artifact_indices] = np.interp(
                        artifact_indices, clean_indices, data[ch, clean_indices]
                    )
        raw._data = data

    def apply_stft(self, data, sampling_rate):
        """Apply STFT to create spectrograms"""
        spectrograms = []
        time_array = None
        for ch in range(data.shape[0]):
            f, t, Zxx = spectrogram(
                data[ch, :],
                fs=sampling_rate,
                nperseg=STFT_NPERSEG,
                noverlap=STFT_NOVERLAP,
                window="hann",
            )
            if time_array is None:
                time_array = t
            freq_mask = (f >= LOW_FREQ_HZ) & (f <= HIGH_FREQ_HZ)
            spectrograms.append(Zxx[freq_mask, :])
        return np.array(spectrograms), f[freq_mask], time_array

    def preprocess_sequences_from_file(
        self, patient_id: str, filename: str, sequences: List[Dict]
    ) -> List[Optional[Dict]]:
        """Process sequences from one file with per-sequence zero-centering."""
        try:
            edf_path = f"physionet.org/files/chbmit/1.0.0/{patient_id}/{filename}"
            # Use preload=False for lazy loading to save memory
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            sampling_rate = raw.info["sfreq"]
            raw_selected, clean_channel_names = self.select_target_channels(raw)

            all_segment_starts = [
                start for seq in sequences for start in seq["segment_starts"]
            ]
            min_time, max_time = (
                min(all_segment_starts),
                max(all_segment_starts) + SEGMENT_DURATION,
            )
            padding = 5.0
            crop_tmin, crop_tmax = max(0, min_time - padding), min(
                raw_selected.times[-1], max_time + padding
            )

            # Crop first, then load only the required data slice into memory
            raw_cropped = raw_selected.copy().crop(tmin=crop_tmin, tmax=crop_tmax)
            raw_cropped.load_data()

            raw_cropped.notch_filter(
                freqs=NOTCH_FREQ_HZ, n_jobs=MNE_N_JOBS, verbose=False
            )
            self.remove_amplitude_artifacts(raw_cropped)

            filtered_data_uv = raw_cropped.get_data() * 1e6
            stft_coeffs, frequencies, time_array = self.apply_stft(
                filtered_data_uv, sampling_rate
            )
            full_power_spectrogram = np.abs(stft_coeffs) ** 2

            processed_sequences = []
            for sequence in sequences:
                try:
                    sequence_spectrograms = []
                    for segment_start in sequence["segment_starts"]:
                        segment_start_relative = segment_start - crop_tmin
                        segment_end_relative = segment_start_relative + SEGMENT_DURATION
                        bin_mask = (time_array >= segment_start_relative) & (
                            time_array < segment_end_relative
                        )
                        power_spectrogram = full_power_spectrogram[:, :, bin_mask]

                        if APPLY_LOG_TRANSFORM:
                            power_spectrogram = np.log10(
                                power_spectrogram + LOG_TRANSFORM_EPSILON
                            )

                        # Per-sequence Z-score Normalization (Mean=0, Std=1)
                        mean = power_spectrogram.mean()
                        std = power_spectrogram.std()
                        if std > 1e-8:
                            power_spectrogram = (power_spectrogram - mean) / std
                        else:
                            power_spectrogram = power_spectrogram - mean

                        sequence_spectrograms.append(power_spectrogram)

                    processed_sequences.append(
                        {
                            "spectrogram": np.stack(sequence_spectrograms, axis=0),
                            "label": (
                                1 if sequence["type"] == self.positive_label else 0
                            ),
                            "patient_id": sequence["patient_id"],
                            "sequence_start_sec": sequence["sequence_start_sec"],
                            "sequence_end_sec": sequence["sequence_end_sec"],
                            "file_name": sequence["file"],
                            "time_to_seizure": sequence.get("time_to_seizure", -1),
                            "channel_names": clean_channel_names,
                            "frequencies": frequencies,
                        }
                    )
                except Exception as e:
                    self.logger.error(f"Sequence error: {e}")
                    processed_sequences.append(None)

            # Explicitly free memory
            del (
                raw,
                raw_selected,
                raw_cropped,
                filtered_data_uv,
                stft_coeffs,
                full_power_spectrogram,
            )

            return processed_sequences
        except Exception as e:
            self.logger.error(f"File error {filename}: {e}")
            return [None] * len(sequences)

    def append_to_hdf5(self, split_name: str, batch_sequences: List[Dict]):
        """Append normalized sequences to HDF5."""
        if not batch_sequences:
            return
        output_file = self.data_dir / f"{split_name}_dataset.h5"
        first_spec = batch_sequences[0]["spectrogram"]
        n_seq_len, n_channels, n_freqs, n_times = first_spec.shape

        with h5py.File(output_file, "a") as f:
            if "spectrograms" not in f:
                f.create_dataset(
                    "spectrograms",
                    (0, n_seq_len, n_channels, n_freqs, n_times),
                    maxshape=(None, n_seq_len, n_channels, n_freqs, n_times),
                    dtype=np.float32,
                    chunks=True,
                )
                f.create_dataset(
                    "labels", (0,), maxshape=(None,), dtype=np.int32, chunks=True
                )
                f.create_dataset(
                    "patient_ids", (0,), maxshape=(None,), dtype="S10", chunks=True
                )
                seg_info = f.create_group("segment_info")
                for k, dt in [
                    ("start_times", np.float32),
                    ("end_times", np.float32),
                    ("file_names", "S20"),
                    ("time_to_seizure", np.float32),
                    ("segment_ids", "S50"),
                ]:
                    seg_info.create_dataset(
                        k, (0,), maxshape=(None,), dtype=dt, chunks=True
                    )
                f.create_group("metadata").attrs[
                    "normalization"
                ] = "per-sequence-zero-centering"

            n_existing = f["spectrograms"].shape[0]
            n_new = len(batch_sequences)
            new_total = n_existing + n_new
            for ds in [f["spectrograms"], f["labels"], f["patient_ids"]] + [
                f["segment_info"][k] for k in f["segment_info"]
            ]:
                ds.resize(new_total, axis=0)

            for i, seq in enumerate(batch_sequences):
                idx = n_existing + i
                f["spectrograms"][idx] = seq["spectrogram"]
                f["labels"][idx] = seq["label"]
                f["patient_ids"][idx] = seq["patient_id"].encode("utf-8")
                si = f["segment_info"]
                si["start_times"][idx] = seq["sequence_start_sec"]
                si["end_times"][idx] = seq["sequence_end_sec"]
                si["file_names"][idx] = seq["file_name"].encode("utf-8")
                si["time_to_seizure"][idx] = seq["time_to_seizure"]
                si["segment_ids"][idx] = (
                    f"{seq['patient_id']}_{seq['sequence_start_sec']}".encode("utf-8")
                )

    def process_and_save_split(self, split_name: str, sequences: List[Dict]):
        """Process and save a full data split with file-level checkpointing."""
        if not sequences:
            return
        self.logger.info(f"Processing {split_name} split: {len(sequences)} sequences")
        file_groups = self.group_sequences_by_file(sequences)

        for (pid, fname), f_seqs in tqdm(
            file_groups.items(), desc=f"Split: {split_name}"
        ):
            # OPTIMIZATION: Checkpoint based on split and filename
            file_id = f"{split_name}/{pid}/{fname}"

            if file_id in self.checkpoint["completed_files"]:
                continue

            processed = self.preprocess_sequences_from_file(pid, fname, f_seqs)

            # Filter valid results
            valid_batch = [p for p in processed if p is not None]

            if valid_batch:
                self.append_to_hdf5(split_name, valid_batch)
                self.checkpoint["processed_count"] += len(valid_batch)

            # Mark file as complete (even if processed results are empty, to prevent infinite retries)
            self.checkpoint["completed_files"].add(file_id)
            self.save_checkpoint()

    def run_preprocessing(self):
        """Main entry point: Process all splits."""
        start_time = time.time()
        with open(self.input_json_path, "r") as f:
            all_seqs = json.load(f)["sequences"]

        splits = {
            s: [seq for seq in all_seqs if seq.get("split") == s]
            for s in ["train", "test"]
        }
        for split_name in ["train", "test"]:
            self.process_and_save_split(split_name, splits[split_name])

        self.logger.info(
            f"Preprocessing complete in {(time.time()-start_time)/60:.1f}m"
        )


if __name__ == "__main__":
    n_folds = len(LOPO_PATIENTS)
    if LOPO_FOLD_ID is None:
        folds_to_process = list(range(n_folds))
    else:
        folds_to_process = [LOPO_FOLD_ID]

    for current_fold in folds_to_process:
        fold_config = get_fold_config(current_fold)
        try:
            preprocessor = EEGPreprocessor(fold_config)
            preprocessor.run_preprocessing()
            print(f"Fold {current_fold} completed!")
        except Exception as e:
            print(f" Error: {e}")
            import traceback

            traceback.print_exc()
