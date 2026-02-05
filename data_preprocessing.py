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
from scipy.signal import spectrogram
from data_segmentation_helpers.config import (
    TASK_MODE,
    SEGMENT_DURATION,
    TARGET_CHANNELS,
    LOW_FREQ_HZ,
    HIGH_FREQ_HZ,
    PHASE_FREQ_BAND,
    AMP_FREQ_BAND,
    NOTCH_FREQ_HZ,
    STFT_NPERSEG,
    STFT_NOVERLAP,
    ARTIFACT_THRESHOLD_STD,
    LOG_TRANSFORM_EPSILON,
    APPLY_LOG_TRANSFORM,
    MNE_N_JOBS,
    PATIENTS,
    PATIENT_INDEX,
    get_patient_config,
)

# Suppress MNE warnings
warnings.filterwarnings("ignore", message="Channel names are not unique")
warnings.filterwarnings("ignore", message=".*duplicates.*", category=RuntimeWarning)
mne.set_log_level("ERROR")


class EEGPreprocessor:
    def __init__(self, patient_config: Dict):
        """Initialize preprocessor with patient-specific configuration."""
        self.patient_id = patient_config["patient_id"]
        output_prefix = patient_config["output_prefix"]
        self.balance_seed = patient_config["random_seed"]

        self.input_json_path = f"{output_prefix}_sequences_{TASK_MODE}.json"

        # Directories
        self.output_dir = Path("preprocessing")
        self.dataset_prefix = output_prefix
        self.data_dir = self.output_dir / "data" / self.dataset_prefix
        self.intermediate_dir = self.output_dir / "intermediate" / self.dataset_prefix
        self.logs_dir = self.output_dir / "logs" / self.dataset_prefix
        self.checkpoint_dir = self.output_dir / "checkpoints" / self.dataset_prefix
        self.checkpoint_file = self.checkpoint_dir / "progress.json"

        self.setup_logging_and_directories()
        self.checkpoint = self.load_checkpoint()

        self.logger.info(f"EEG Preprocessor initialized for {self.patient_id}")
        self.logger.info(
            "Strategy: Dual-Stream (Phase + Amp) with Global Normalization"
        )

    @property
    def positive_label(self):
        """Get positive class label based on task mode"""
        return "preictal" if TASK_MODE == "prediction" else "ictal"

    @staticmethod
    def group_sequences_by_file(
        sequences: List[Dict],
    ) -> Dict[Tuple[str, str], List[Dict]]:
        from collections import defaultdict

        file_groups = defaultdict(list)
        for sequence in sequences:
            key = (sequence["patient_id"], sequence["file"])
            file_groups[key].append(sequence)
        return dict(file_groups)

    def setup_logging_and_directories(self):
        for dir_path in [
            self.data_dir,
            self.intermediate_dir,
            self.logs_dir,
            self.checkpoint_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        log_file = self.logs_dir / "preprocessing.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file, mode="a"), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def load_checkpoint(self) -> Dict:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    checkpoint = json.load(f)
                checkpoint["completed_files"] = set(
                    checkpoint.get("completed_files", [])
                )
                return checkpoint
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint: {e}")
        return {"completed_files": set(), "processed_count": 0}

    def save_checkpoint(self):
        checkpoint_copy = self.checkpoint.copy()
        checkpoint_copy["completed_files"] = list(checkpoint_copy["completed_files"])
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint_copy, f, indent=2)

    def select_target_channels(self, raw):
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
        return raw.copy().pick(available_channels), clean_channel_names

    def remove_amplitude_artifacts(self, raw):
        data = raw.get_data()
        for ch in range(data.shape[0]):
            ch_data = data[ch, :]
            median_val = np.median(ch_data)
            mad = np.median(np.abs(ch_data - median_val))
            threshold = ARTIFACT_THRESHOLD_STD * mad * 1.4826
            mask = np.abs(ch_data - median_val) > threshold
            if np.any(mask):
                indices = np.where(mask)[0]
                clean = np.where(~mask)[0]
                if len(clean) > 10:
                    data[ch, indices] = np.interp(indices, clean, data[ch, clean])
        raw._data = data

    def apply_stft(self, data, sampling_rate):
        """Apply STFT with high spectral resolution (nfft=2560 for 0.1Hz spacing)."""
        spectrograms = []
        time_array = None
        freq_array = None

        # nfft = sampling_rate / desired_resolution = 256 / 0.1 = 2560
        nfft_val = int(sampling_rate / 0.1)

        for ch in range(data.shape[0]):
            f, t, Zxx = spectrogram(
                data[ch, :],
                fs=sampling_rate,
                nperseg=STFT_NPERSEG,
                noverlap=STFT_NOVERLAP,
                nfft=nfft_val,  # Force 0.1 Hz frequency bins
                window="hann",
                mode="complex",
            )
            if time_array is None:
                time_array = t
            if freq_array is None:
                freq_array = f
            spectrograms.append(Zxx)

        return np.array(spectrograms), freq_array, time_array

    def extract_spectrograms_from_file(
        self, patient_id: str, filename: str, sequences: List[Dict]
    ) -> List[Optional[Dict]]:
        """
        Pass 1 (Optimized):
        1. Identify ALL unique segments required by these sequences.
        2. Compute STFT for each unique segment ONCE.
        3. Reassemble sequences from these pre-computed blocks.
        """
        try:
            # 1. Collect all unique segment start times required
            # Using a set to remove duplicates automatically
            unique_starts = set()
            for seq in sequences:
                unique_starts.update(seq["segment_starts"])

            # Sort them for efficient file reading (sequential access is faster)
            sorted_starts = sorted(list(unique_starts))

            # 2. Load Raw Data
            edf_path = f"physionet.org/files/chbmit/1.0.0/{patient_id}/{filename}"
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            sampling_rate = raw.info["sfreq"]
            raw_selected, clean_channels = self.select_target_channels(raw)

            # Determine crop bounds
            crop_tmin, crop_tmax = max(0, min(sorted_starts) - 5.0), min(
                raw_selected.times[-1], max(sorted_starts) + SEGMENT_DURATION + 5.0
            )

            # Load Data into RAM
            raw_cropped = (
                raw_selected.copy().crop(tmin=crop_tmin, tmax=crop_tmax).load_data()
            )
            raw_cropped.notch_filter(
                freqs=NOTCH_FREQ_HZ, n_jobs=MNE_N_JOBS, verbose=False
            )
            self.remove_amplitude_artifacts(raw_cropped)
            data_uv = raw_cropped.get_data() * 1e6

            # 3. Pre-compute frequency masks (Fast)
            nfft_val = int(sampling_rate / 0.1)
            freqs = np.fft.rfftfreq(nfft_val, d=1 / sampling_rate)
            phase_mask = (freqs >= PHASE_FREQ_BAND[0]) & (freqs <= PHASE_FREQ_BAND[1])
            amp_mask = (freqs >= AMP_FREQ_BAND[0]) & (freqs <= AMP_FREQ_BAND[1])

            # 4. Compute & Cache Unique Segments
            # Dictionary: start_time -> { 'phase': np.array, 'amp': np.array }
            segment_cache = {}

            for start in tqdm(
                sorted_starts, desc=f"Processing segments ({filename})", leave=False
            ):
                # Calculate indices
                start_rel = start - crop_tmin
                start_idx = int(start_rel * sampling_rate)
                end_idx = start_idx + int(SEGMENT_DURATION * sampling_rate)

                if end_idx > data_uv.shape[1]:
                    continue

                chunk = data_uv[:, start_idx:end_idx]

                # STFT
                z_chunk, f_chunk, t_chunk = self.apply_stft(chunk, sampling_rate)

                # Phase (Sin/Cos Transform)
                z_phase = z_chunk[:, phase_mask, :]
                angles = np.angle(z_phase)
                phase_sin = np.sin(angles)
                phase_cos = np.cos(angles)
                # Concatenate along the channel axis (axis 0)
                # Resulting shape: (num_channels * 2, freqs, time)
                phase_map = np.concatenate([phase_sin, phase_cos], axis=0)

                # Amp
                z_amp_full = z_chunk[:, amp_mask, :]
                z_amp = z_amp_full[:, ::10, :]  # Downsample freq
                amp_map = np.abs(z_amp) ** 2
                if APPLY_LOG_TRANSFORM:
                    amp_map = np.log10(amp_map + LOG_TRANSFORM_EPSILON)

                segment_cache[start] = {"phase": phase_map, "amp": amp_map}

            # 5. Reassemble Sequences
            results = []
            for seq in sequences:
                seq_phase_list = []
                seq_amp_list = []
                valid = True

                for start in seq["segment_starts"]:
                    if start in segment_cache:
                        seg_data = segment_cache[start]
                        seq_phase_list.append(seg_data["phase"])
                        seq_amp_list.append(seg_data["amp"])
                    else:
                        valid = False
                        break

                if not valid or not seq_phase_list:
                    results.append(None)
                    continue

                # Pad/Trim time dimensions if inconsistent
                min_t = min(x.shape[2] for x in seq_phase_list)
                seq_phase_stack = np.stack(
                    [x[:, :, :min_t] for x in seq_phase_list], axis=0
                )
                seq_amp_stack = np.stack(
                    [x[:, :, :min_t] for x in seq_amp_list], axis=0
                )

                results.append(
                    {
                        "spectrogram_phase": seq_phase_stack,
                        "spectrogram_amp": seq_amp_stack,
                        "label": 1 if seq["type"] == self.positive_label else 0,
                        "patient_id": patient_id,
                        "file_name": filename,
                        "start_sec": seq["sequence_start_sec"],
                        "time_to_seizure": seq.get("time_to_seizure", -1),
                    }
                )

            return results

        except Exception as e:
            self.logger.error(f"Error processing file {filename}: {e}")
            import traceback

            traceback.print_exc()
            return [None] * len(sequences)

    def _init_hdf5(self, path, phase_shape, amp_shape):
        with h5py.File(path, "w") as f:
            # Dual Datasets
            f.create_dataset(
                "spectrograms_phase",
                (0, *phase_shape),
                maxshape=(None, *phase_shape),
                dtype=np.float32,
                chunks=True,
            )
            f.create_dataset(
                "spectrograms_amp",
                (0, *amp_shape),
                maxshape=(None, *amp_shape),
                dtype=np.float32,
                chunks=True,
            )

            f.create_dataset("labels", (0,), maxshape=(None,), dtype=np.int32)
            f.create_dataset("patient_ids", (0,), maxshape=(None,), dtype="S10")
            si = f.create_group("segment_info")
            for k, dt in [
                ("start_times", np.float32),
                ("file_names", "S20"),
                ("time_to_seizure", np.float32),
            ]:
                si.create_dataset(k, (0,), maxshape=(None,), dtype=dt)

    def _append_to_hdf5(self, path, batch):
        with h5py.File(path, "a") as f:
            n_old = f["spectrograms_phase"].shape[0]
            n_new = len(batch)

            # Resize all
            for ds_name in [
                "spectrograms_phase",
                "spectrograms_amp",
                "labels",
                "patient_ids",
            ]:
                f[ds_name].resize(n_old + n_new, axis=0)
            for k in ["start_times", "file_names", "time_to_seizure"]:
                f[f"segment_info/{k}"].resize(n_old + n_new, axis=0)

            for i, item in enumerate(batch):
                idx = n_old + i
                f["spectrograms_phase"][idx] = item["spectrogram_phase"]
                f["spectrograms_amp"][idx] = item["spectrogram_amp"]
                f["labels"][idx] = item["label"]
                f["patient_ids"][idx] = item["patient_id"].encode("utf-8")
                f["segment_info/start_times"][idx] = item["start_sec"]
                f["segment_info/file_names"][idx] = item["file_name"].encode("utf-8")
                f["segment_info/time_to_seizure"][idx] = item["time_to_seizure"]

    def create_intermediate_datasets(self, splits):
        for split, sequences in splits.items():
            path = self.intermediate_dir / f"s{split}_intermediate.h5"

            if path.exists() and f"pass1_{split}_complete" in self.checkpoint:
                continue

            groups = self.group_sequences_by_file(sequences)
            for (pid, fname), f_seqs in tqdm(
                groups.items(), desc=f"Pass 1 (Dual-Stream): {split}"
            ):
                file_id = f"p1_{split}_{fname}"
                if file_id in self.checkpoint["completed_files"]:
                    continue

                batch = [
                    b
                    for b in self.extract_spectrograms_from_file(pid, fname, f_seqs)
                    if b is not None
                ]
                if batch:
                    if not path.exists():
                        self._init_hdf5(
                            path,
                            batch[0]["spectrogram_phase"].shape,
                            batch[0]["spectrogram_amp"].shape,
                        )
                    self._append_to_hdf5(path, batch)
                self.checkpoint["completed_files"].add(file_id)
                self.save_checkpoint()
            self.checkpoint[f"pass1_{split}_complete"] = True
            self.save_checkpoint()

    def compute_stats(self, split_names) -> Tuple[float, float]:
        self.logger.info("Pass 2: Computing global stats (Amplitude Stream ONLY)...")
        
        sum_v, sum_sq, count = 0.0, 0.0, 0
        # path = self.intermediate_dir / "train_intermediate.h5"
        for split in split_names:
            path = self.intermediate_dir / f"s{split}_intermediate.h5"
            with h5py.File(path, "r") as f:
                ds = f["spectrograms_amp"]  # Only normalize amplitude
                for i in tqdm(range(0, ds.shape[0], 100), desc="Stats"):
                    chunk = ds[i : i + 100]
                    sum_v += np.sum(chunk)
                    sum_sq += np.sum(chunk**2)
                    count += chunk.size
        # mean = sum_v / count
        # std = np.sqrt((sum_sq / count) - (mean**2))
        # return float(mean), float(std)
        return float(sum_v), float(sum_sq), float(count)

    def create_final_datasets(self, mean, std, split_names):
        for split in split_names:
            src, dst = (
                self.intermediate_dir / f"s{split}_intermediate.h5",
                self.data_dir / f"s{split}_dataset.h5",
            )
            if not src.exists():
                continue
            self.logger.info(f"Pass 3: Finalizing split {split} (Norm Amp, Raw Phase)...")
            with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
                # Copy Metadata / Labels
                for k in ["labels", "patient_ids"]:
                    fout.create_dataset(k, data=fin[k][:])
                gi, go = fin["segment_info"], fout.create_group("segment_info")
                for k in gi:
                    go.create_dataset(k, data=gi[k][:])

                # Setup Datasets
                phase_in = fin["spectrograms_phase"]
                amp_in = fin["spectrograms_amp"]

                # Phase: Direct Copy (No Normalization)
                phase_out = fout.create_dataset(
                    "spectrograms_phase", phase_in.shape, dtype=np.float32
                )
                # Amp: Normalized Copy
                amp_out = fout.create_dataset(
                    "spectrograms_amp", amp_in.shape, dtype=np.float32
                )

                chunk_size = 100
                total = phase_in.shape[0]

                for i in tqdm(range(0, total, chunk_size), desc=f"Final {split}"):
                    # Copy Phase
                    phase_out[i : i + chunk_size] = phase_in[i : i + chunk_size]

                    # Normalize Amp
                    chunk_amp = amp_in[i : i + chunk_size]
                    amp_out[i : i + chunk_size] = (
                        (chunk_amp - mean) / std if std > 1e-8 else chunk_amp - mean
                    )

                fout.create_group("metadata").attrs.update(
                    {"mean": mean, "std": std, "norm": "dual_stream_amp_only"}
                )

    def run_preprocessing(self):
        with open(self.input_json_path, "r") as f:
            all_seqs = json.load(f)["sequences"]
        split_names = set([seq.get("split") for seq in all_seqs])
        splits = {
            s: [seq for seq in all_seqs if seq.get("split") == s]
            for s in split_names
        }

        self.create_intermediate_datasets(splits)

        if "patient_stats" not in self.checkpoint:
            self.checkpoint.update({"patient_stats": {}})
        
        if self.patient_id not in self.checkpoint["patient_stats"]:
            # mean, std = self.compute_stats(split_names)
            sum_v, sum_sq, count = self.compute_stats(split_names)
            self.checkpoint["patient_stats"].update(
                {"sum_v": sum_v, "sum_sq": sum_sq, "count": count})
            self.save_checkpoint()
        else:
            # mean, std = self.checkpoint["global_mean"], self.checkpoint["global_std"]
            sum_v = self.checkpoint["patient_stats"]["sum_v"]
            sum_sq = self.checkpoint["patient_stats"]["sum_sq"]
            count = self.checkpoint["patient_stats"]["count"]

        mean = float(sum_v / count)
        std = float( np.sqrt((sum_sq / count) - (mean**2))  )
        self.create_final_datasets(mean, std, split_names)
        self.logger.info("Preprocessing complete.")


if __name__ == "__main__":
    for i in range(len(PATIENTS)):
        if PATIENT_INDEX is not None and i != PATIENT_INDEX:
            continue
        cfg = get_patient_config(i)
        try:
            EEGPreprocessor(cfg).run_preprocessing()
        except Exception as e:
            print(f"Error {cfg['patient_id']}: {e}")
    
    

