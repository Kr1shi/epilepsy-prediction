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
        self.logger.info("Strategy: 3-Pass Global Normalization (Efficient)")

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

    def extract_spectrograms_from_file(
        self, patient_id: str, filename: str, sequences: List[Dict]
    ) -> List[Optional[Dict]]:
        """Pass 1 helper: Load, filter, compute STFT, and log-transform."""
        try:
            edf_path = f"physionet.org/files/chbmit/1.0.0/{patient_id}/{filename}"
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            sampling_rate = raw.info["sfreq"]
            raw_selected, clean_channels = self.select_target_channels(raw)

            all_starts = [s for seq in sequences for s in seq["segment_starts"]]
            crop_tmin, crop_tmax = max(0, min(all_starts) - 5.0), min(
                raw_selected.times[-1], max(all_starts) + SEGMENT_DURATION + 5.0
            )

            raw_cropped = (
                raw_selected.copy().crop(tmin=crop_tmin, tmax=crop_tmax).load_data()
            )
            raw_cropped.notch_filter(
                freqs=NOTCH_FREQ_HZ, n_jobs=MNE_N_JOBS, verbose=False
            )
            self.remove_amplitude_artifacts(raw_cropped)

            data_uv = raw_cropped.get_data() * 1e6
            stft_coeffs, frequencies, time_arr = self.apply_stft(data_uv, sampling_rate)
            full_power = np.abs(stft_coeffs) ** 2

            results = []
            for seq in sequences:
                seq_specs = []
                for start in seq["segment_starts"]:
                    start_rel = start - crop_tmin
                    bin_mask = (time_arr >= start_rel) & (
                        time_arr < start_rel + SEGMENT_DURATION
                    )
                    spec = full_power[:, :, bin_mask]
                    if APPLY_LOG_TRANSFORM:
                        spec = np.log10(spec + LOG_TRANSFORM_EPSILON)
                    seq_specs.append(spec)

                results.append(
                    {
                        "spectrogram": np.stack(seq_specs, axis=0),
                        "label": 1 if seq["type"] == self.positive_label else 0,
                        "patient_id": patient_id,
                        "file_name": filename,
                        "start_sec": seq["sequence_start_sec"],
                        "time_to_seizure": seq.get("time_to_seizure", -1),
                    }
                )
            return results
        except Exception as e:
            self.logger.error(f"Error extracting {filename}: {e}")
            return [None] * len(sequences)

    def _init_hdf5(self, path, shape):
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "spectrograms",
                (0, *shape),
                maxshape=(None, *shape),
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
            n_old = f["spectrograms"].shape[0]
            n_new = len(batch)
            for ds in [
                f["spectrograms"],
                f["labels"],
                f["patient_ids"],
                f["segment_info/start_times"],
                f["segment_info/file_names"],
                f["segment_info/time_to_seizure"],
            ]:
                ds.resize(n_old + n_new, axis=0)
            for i, item in enumerate(batch):
                idx = n_old + i
                f["spectrograms"][idx] = item["spectrogram"]
                f["labels"][idx] = item["label"]
                f["patient_ids"][idx] = item["patient_id"].encode("utf-8")
                f["segment_info/start_times"][idx] = item["start_sec"]
                f["segment_info/file_names"][idx] = item["file_name"].encode("utf-8")
                f["segment_info/time_to_seizure"][idx] = item["time_to_seizure"]

    def create_intermediate_datasets(self, splits):
        for split, sequences in splits.items():
            path = self.intermediate_dir / f"{split}_intermediate.h5"
            if path.exists() and f"pass1_{split}_complete" in self.checkpoint:
                continue

            groups = self.group_sequences_by_file(sequences)
            for (pid, fname), f_seqs in tqdm(groups.items(), desc=f"Pass 1: {split}"):
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
                        self._init_hdf5(path, batch[0]["spectrogram"].shape)
                    self._append_to_hdf5(path, batch)
                self.checkpoint["completed_files"].add(file_id)
                self.save_checkpoint()
            self.checkpoint[f"pass1_{split}_complete"] = True
            self.save_checkpoint()

    def compute_stats(self) -> Tuple[float, float]:
        path = self.intermediate_dir / "train_intermediate.h5"
        self.logger.info("Pass 2: Computing global stats...")
        with h5py.File(path, "r") as f:
            ds = f["spectrograms"]
            sum_v, sum_sq, count = 0.0, 0.0, 0
            for i in tqdm(range(0, ds.shape[0], 100), desc="Stats"):
                chunk = ds[i : i + 100]
                sum_v += np.sum(chunk)
                sum_sq += np.sum(chunk**2)
                count += chunk.size
        mean = sum_v / count
        std = np.sqrt((sum_sq / count) - (mean**2))
        return float(mean), float(std)

    def create_final_datasets(self, mean, std):
        for split in ["train", "test"]:
            src, dst = (
                self.intermediate_dir / f"{split}_intermediate.h5",
                self.data_dir / f"{split}_dataset.h5",
            )
            if not src.exists():
                continue
            self.logger.info(f"Pass 3: Normalizing {split}...")
            with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
                for k in ["labels", "patient_ids"]:
                    fout.create_dataset(k, data=fin[k][:])
                gi, go = fin["segment_info"], fout.create_group("segment_info")
                for k in gi:
                    go.create_dataset(k, data=gi[k][:])

                ds_in = fin["spectrograms"]
                ds_out = fout.create_dataset(
                    "spectrograms", ds_in.shape, dtype=np.float32
                )
                for i in tqdm(range(0, ds_in.shape[0], 100), desc=f"Final {split}"):
                    chunk = ds_in[i : i + 100]
                    ds_out[i : i + 100] = (
                        (chunk - mean) / std if std > 1e-8 else chunk - mean
                    )

                fout.create_group("metadata").attrs.update(
                    {"mean": mean, "std": std, "norm": "global"}
                )

    def run_preprocessing(self):
        with open(self.input_json_path, "r") as f:
            all_seqs = json.load(f)["sequences"]
        splits = {
            s: [seq for seq in all_seqs if seq.get("split") == s]
            for s in ["train", "test"]
        }

        self.create_intermediate_datasets(splits)

        if "global_mean" not in self.checkpoint:
            mean, std = self.compute_stats()
            self.checkpoint.update({"global_mean": mean, "global_std": std})
            self.save_checkpoint()
        else:
            mean, std = self.checkpoint["global_mean"], self.checkpoint["global_std"]

        self.create_final_datasets(mean, std)
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
