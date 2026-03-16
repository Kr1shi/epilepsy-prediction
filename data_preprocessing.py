import mne
import numpy as np
import h5py
import json
import logging
import warnings
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from scipy.signal import spectrogram
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_segmentation_helpers.config import (
    TASK_MODE,
    SEGMENT_DURATION,
    TARGET_CHANNELS,
    FULL_FREQ_BAND,
    NOTCH_FREQ_HZ,
    STFT_NPERSEG,
    STFT_NOVERLAP,
    STFT_NFFT,
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
warnings.filterwarnings("ignore", message=".*Scaling factor.*", category=RuntimeWarning)
mne.set_log_level("ERROR")

# Pre-compute frequency mask (depends only on STFT_NFFT and assumed 256 Hz sampling rate)
_ASSUMED_SFREQ = 256.0
_FREQS = np.fft.rfftfreq(STFT_NFFT, d=1 / _ASSUMED_SFREQ)
FREQ_MASK = (_FREQS >= FULL_FREQ_BAND[0]) & (_FREQS <= FULL_FREQ_BAND[1])

# Max workers for parallel file processing (leave some cores for MNE's own parallelism)
MAX_WORKERS = max(1, 16 // max(MNE_N_JOBS, 1))


def _select_target_channels(raw):
    """Select and rename target channels from raw EDF."""
    available_channels = []
    for ch in TARGET_CHANNELS:
        if ch in raw.ch_names:
            available_channels.append(ch)
        else:
            matches = [n for n in raw.ch_names if n.startswith(ch + "-")]
            if matches:
                available_channels.append(matches[0])
    return raw.copy().pick(available_channels)


def _remove_amplitude_artifacts(data):
    """Vectorized MAD-based artifact detection and interpolation.

    Args:
        data: (n_channels, n_samples) array, modified in-place.
    """
    # Compute per-channel median and MAD
    medians = np.median(data, axis=1, keepdims=True)  # (C, 1)
    deviations = np.abs(data - medians)
    mads = np.median(deviations, axis=1, keepdims=True)  # (C, 1)
    thresholds = ARTIFACT_THRESHOLD_STD * mads * 1.4826

    # Boolean mask of artifacts: (C, N)
    artifact_mask = deviations > thresholds

    # Interpolate per-channel (np.interp requires per-channel calls)
    for ch in range(data.shape[0]):
        if not np.any(artifact_mask[ch]):
            continue
        bad_idx = np.where(artifact_mask[ch])[0]
        good_idx = np.where(~artifact_mask[ch])[0]
        if len(good_idx) > 10:
            data[ch, bad_idx] = np.interp(bad_idx, good_idx, data[ch, good_idx])


def _apply_stft(data, sampling_rate):
    """Apply STFT to all channels at once."""
    _, _, Zxx = spectrogram(
        data,
        fs=sampling_rate,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        nfft=STFT_NFFT,
        window="hann",
        mode="complex",
        axis=-1,
    )
    return Zxx


def process_single_file(patient_id, filename, sequences, positive_label):
    """Process one EDF file: load, filter, STFT, assemble sequences.

    This is a standalone function so it can be called from ProcessPoolExecutor.

    Returns:
        List of result dicts (or None for failed sequences), each containing
        'spectrogram', 'label', 'patient_id', 'file_name', 'start_sec',
        'time_to_seizure'.
    """
    try:
        # 1. Collect all unique segment start times
        unique_starts = set()
        for seq in sequences:
            unique_starts.update(seq["segment_starts"])
        sorted_starts = sorted(unique_starts)

        # 2. Load Raw Data
        edf_path = f"physionet.org/files/chbmit/1.0.0/{patient_id}/{filename}"
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        sampling_rate = raw.info["sfreq"]
        raw_selected = _select_target_channels(raw)

        # Determine crop bounds
        crop_tmin = max(0, min(sorted_starts) - 5.0)
        crop_tmax = min(
            raw_selected.times[-1], max(sorted_starts) + SEGMENT_DURATION + 5.0
        )

        # Load & filter
        raw_cropped = raw_selected.crop(tmin=crop_tmin, tmax=crop_tmax).load_data()
        raw_cropped.notch_filter(
            freqs=NOTCH_FREQ_HZ, n_jobs=MNE_N_JOBS, verbose=False
        )

        data = raw_cropped.get_data()
        _remove_amplitude_artifacts(data)
        data_uv = data * 1e6

        # Use pre-computed freq mask, but verify sampling rate matches
        if sampling_rate != _ASSUMED_SFREQ:
            freqs = np.fft.rfftfreq(STFT_NFFT, d=1 / sampling_rate)
            freq_mask = (freqs >= FULL_FREQ_BAND[0]) & (freqs <= FULL_FREQ_BAND[1])
        else:
            freq_mask = FREQ_MASK

        # 3. Compute & Cache Unique Segments
        segment_cache = {}
        seg_duration_samples = int(SEGMENT_DURATION * sampling_rate)

        for start in sorted_starts:
            start_rel = start - crop_tmin
            start_idx = int(start_rel * sampling_rate)
            end_idx = start_idx + seg_duration_samples

            if end_idx > data_uv.shape[1]:
                continue

            chunk = data_uv[:, start_idx:end_idx]
            z_chunk = _apply_stft(chunk, sampling_rate)

            power_map = np.abs(z_chunk[:, freq_mask, :]) ** 2
            if APPLY_LOG_TRANSFORM:
                power_map = np.log10(power_map + LOG_TRANSFORM_EPSILON)

            segment_cache[start] = power_map

        # 4. Reassemble Sequences
        results = []
        for seq in sequences:
            spec_list = []
            valid = True

            for start in seq["segment_starts"]:
                if start in segment_cache:
                    spec_list.append(segment_cache[start])
                else:
                    valid = False
                    break

            if not valid or not spec_list:
                results.append(None)
                continue

            min_t = min(x.shape[2] for x in spec_list)
            spec_stack = np.stack(
                [x[:, :, :min_t] for x in spec_list], axis=0
            )

            results.append(
                {
                    "spectrogram": spec_stack,
                    "label": 1 if seq["type"] == positive_label else 0,
                    "patient_id": patient_id,
                    "file_name": filename,
                    "start_sec": seq["sequence_start_sec"],
                    "time_to_seizure": seq.get("time_to_seizure", -1),
                }
            )

        return results

    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        import traceback
        traceback.print_exc()
        return [None] * len(sequences)


class EEGPreprocessor:
    def __init__(self, patient_config: Dict):
        """Initialize preprocessor with patient-specific configuration."""
        self.patient_id = patient_config["patient_id"]
        output_prefix = patient_config["output_prefix"]

        self.input_json_path = f"{output_prefix}_sequences_{TASK_MODE}.json"

        # Directories
        self.output_dir = Path("preprocessing")
        self.dataset_prefix = output_prefix
        self.data_dir = self.output_dir / "data" / self.dataset_prefix
        self.logs_dir = self.output_dir / "logs" / self.dataset_prefix
        self.checkpoint_dir = self.output_dir / "checkpoints" / self.dataset_prefix
        self.checkpoint_file = self.checkpoint_dir / "progress.json"

        self.setup_logging_and_directories()
        self.checkpoint = self.load_checkpoint()

        self.logger.info(f"EEG Preprocessor initialized for {self.patient_id}")

    @property
    def positive_label(self):
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

    def _init_hdf5(self, path, spec_shape):
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "spectrograms",
                (0, *spec_shape),
                maxshape=(None, *spec_shape),
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

            for ds_name in ["spectrograms", "labels", "patient_ids"]:
                f[ds_name].resize(n_old + n_new, axis=0)
            for k in ["start_times", "file_names", "time_to_seizure"]:
                f[f"segment_info/{k}"].resize(n_old + n_new, axis=0)

            # Batch write
            specs = np.stack([item["spectrogram"] for item in batch], axis=0)
            f["spectrograms"][n_old:n_old + n_new] = specs
            f["labels"][n_old:n_old + n_new] = [item["label"] for item in batch]
            f["patient_ids"][n_old:n_old + n_new] = [
                item["patient_id"].encode("utf-8") for item in batch
            ]
            f["segment_info/start_times"][n_old:n_old + n_new] = [
                item["start_sec"] for item in batch
            ]
            f["segment_info/file_names"][n_old:n_old + n_new] = [
                item["file_name"].encode("utf-8") for item in batch
            ]
            f["segment_info/time_to_seizure"][n_old:n_old + n_new] = [
                item["time_to_seizure"] for item in batch
            ]

    def run_preprocessing(self):
        """Single-pass preprocessing: process files in parallel, accumulate stats,
        write unnormalized HDF5, then normalize in a final pass."""

        with open(self.input_json_path, "r") as f:
            all_seqs = json.load(f)["sequences"]
        split_names = set(seq.get("split") for seq in all_seqs)
        splits = {
            s: [seq for seq in all_seqs if seq.get("split") == s]
            for s in split_names
        }

        # ── Pass 1: Process EDF files → unnormalized HDF5 + running stats ──
        # Accumulate Welford running stats across all splits
        sum_v, sum_sq, count = 0.0, 0.0, 0
        stats_from_checkpoint = False

        if "patient_stats" in self.checkpoint and self.patient_id in self.checkpoint.get("patient_stats", {}):
            sum_v = self.checkpoint["patient_stats"]["sum_v"]
            sum_sq = self.checkpoint["patient_stats"]["sum_sq"]
            count = self.checkpoint["patient_stats"]["count"]
            stats_from_checkpoint = True

        for split, sequences in splits.items():
            unnorm_path = self.data_dir / f"s{split}_unnorm.h5"

            if f"pass1_{split}_complete" in self.checkpoint:
                continue

            groups = self.group_sequences_by_file(sequences)

            # Filter out already-completed files
            pending = {
                k: v for k, v in groups.items()
                if f"p1_{split}_{k[1]}" not in self.checkpoint["completed_files"]
            }

            if not pending:
                self.checkpoint[f"pass1_{split}_complete"] = True
                self.save_checkpoint()
                continue

            # Process files in parallel
            futures = {}
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for (pid, fname), f_seqs in pending.items():
                    future = executor.submit(
                        process_single_file, pid, fname, f_seqs, self.positive_label
                    )
                    futures[future] = (pid, fname, split)

                pbar = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Pass 1: {split}",
                )
                for future in pbar:
                    pid, fname, split_name = futures[future]
                    file_id = f"p1_{split_name}_{fname}"

                    results = future.result()
                    batch = [r for r in results if r is not None]

                    if batch:
                        if not unnorm_path.exists():
                            self._init_hdf5(unnorm_path, batch[0]["spectrogram"].shape)
                        self._append_to_hdf5(unnorm_path, batch)

                        # Accumulate stats from this batch
                        if not stats_from_checkpoint:
                            for item in batch:
                                spec = item["spectrogram"]
                                sum_v += np.sum(spec)
                                sum_sq += np.sum(spec ** 2)
                                count += spec.size

                    self.checkpoint["completed_files"].add(file_id)
                    pbar.set_postfix({"file": fname})

            self.checkpoint[f"pass1_{split}_complete"] = True
            self.save_checkpoint()

        # Save stats
        if not stats_from_checkpoint and count > 0:
            if "patient_stats" not in self.checkpoint:
                self.checkpoint["patient_stats"] = {}
            self.checkpoint["patient_stats"].update(
                {"sum_v": float(sum_v), "sum_sq": float(sum_sq), "count": float(count)}
            )
            self.save_checkpoint()

        if count == 0:
            self.logger.error("No data processed — cannot compute normalization stats.")
            return

        mean = float(sum_v / count)
        std = float(np.sqrt((sum_sq / count) - (mean ** 2)))
        self.logger.info(f"Normalization stats: mean={mean:.6f}, std={std:.6f}")

        # ── Pass 2: Normalize unnormalized HDF5 → final dataset ──
        for split in split_names:
            unnorm_path = self.data_dir / f"s{split}_unnorm.h5"
            final_path = self.data_dir / f"s{split}_dataset.h5"

            if not unnorm_path.exists():
                continue
            if final_path.exists() and f"final_{split}_complete" in self.checkpoint:
                continue

            self.logger.info(f"Pass 2: Normalizing split {split}...")
            with h5py.File(unnorm_path, "r") as fin, h5py.File(final_path, "w") as fout:
                # Copy metadata
                for k in ["labels", "patient_ids"]:
                    fout.create_dataset(k, data=fin[k][:])
                gi, go = fin["segment_info"], fout.create_group("segment_info")
                for k in gi:
                    go.create_dataset(k, data=gi[k][:])

                # Normalize spectrograms in chunks
                spec_in = fin["spectrograms"]
                spec_out = fout.create_dataset(
                    "spectrograms", spec_in.shape, dtype=np.float32
                )

                chunk_size = 100
                for i in tqdm(range(0, spec_in.shape[0], chunk_size), desc=f"Normalize {split}"):
                    chunk = spec_in[i : i + chunk_size]
                    spec_out[i : i + chunk_size] = (
                        (chunk - mean) / std if std > 1e-8 else chunk - mean
                    )

                fout.create_group("metadata").attrs.update(
                    {"mean": mean, "std": std, "norm": "single_stream_zscore"}
                )

            self.checkpoint[f"final_{split}_complete"] = True
            self.save_checkpoint()

            # Clean up unnormalized file
            unnorm_path.unlink()

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
