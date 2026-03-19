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


def process_single_file(patient_id, filename, sequences, positive_label):
    """Process one EDF file: load, filter, STFT, assemble sequences.

    This is a standalone function so it can be called from ProcessPoolExecutor.
    Computes a single STFT over the entire cropped region and extracts segments
    by index slicing, avoiding redundant FFT calls for overlapping segments.

    Returns:
        Tuple of (results, stats) where:
        - results: list of result dicts (or None for failed sequences)
        - stats: tuple of (sum_v, sum_sq, count) running statistics
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

        # 3. Compute STFT once over the entire cropped data
        _, t_stft, Zxx_full = spectrogram(
            data_uv,
            fs=sampling_rate,
            nperseg=STFT_NPERSEG,
            noverlap=STFT_NOVERLAP,
            nfft=STFT_NFFT,
            window="hann",
            mode="complex",
            axis=-1,
        )
        # Apply freq mask, compute power, log transform → float32
        power_full = np.abs(Zxx_full[:, freq_mask, :]) ** 2
        if APPLY_LOG_TRANSFORM:
            power_full = np.log10(power_full + LOG_TRANSFORM_EPSILON)
        power_full = power_full.astype(np.float32)

        # Pre-compute how many STFT bins fall in one segment
        hop = STFT_NPERSEG - STFT_NOVERLAP
        seg_duration_samples = int(SEGMENT_DURATION * sampling_rate)
        seg_n_bins = (seg_duration_samples - STFT_NPERSEG) // hop + 1
        half_win_sec = STFT_NPERSEG / (2 * sampling_rate)

        # 4. Extract cached segments by slicing from the full STFT
        segment_cache = {}
        for start in sorted_starts:
            start_rel = start - crop_tmin
            # First STFT bin center within this segment
            first_bin_time = start_rel + half_win_sec
            i_start = np.searchsorted(t_stft, first_bin_time - 1e-6)
            i_end = i_start + seg_n_bins

            if i_end <= power_full.shape[2]:
                segment_cache[start] = power_full[:, :, i_start:i_end]

        # Free the large full STFT arrays
        del Zxx_full, power_full

        # 5. Reassemble Sequences and accumulate running stats
        results = []
        batch_sum_v = np.float64(0.0)
        batch_sum_sq = np.float64(0.0)
        batch_count = 0

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

            # Accumulate running stats in worker (avoids main-thread loop)
            batch_sum_v += np.sum(spec_stack, dtype=np.float64)
            batch_sum_sq += np.sum(spec_stack.astype(np.float64) ** 2)
            batch_count += spec_stack.size

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

        return results, (float(batch_sum_v), float(batch_sum_sq), batch_count)

    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        import traceback
        traceback.print_exc()
        return [None] * len(sequences), (0.0, 0.0, 0)


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
        """Process EDF files → HDF5 with per-split statistics.

        Data is saved WITHOUT normalization. Per-split running statistics
        (sum, sum_sq, count) are stored in HDF5 metadata so that training
        code can compute normalization from training splits only, preventing
        data leakage from test splits."""

        with open(self.input_json_path, "r") as f:
            all_seqs = json.load(f)["sequences"]
        split_names = set(seq.get("split") for seq in all_seqs)
        splits = {
            s: [seq for seq in all_seqs if seq.get("split") == s]
            for s in split_names
        }

        # Single executor shared across all splits — avoids per-split fork + import overhead
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for split, sequences in splits.items():
                final_path = self.data_dir / f"s{split}_dataset.h5"

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

                # Per-split running stats (for train-only normalization at training time)
                sum_v, sum_sq, count = 0.0, 0.0, 0

                # Submit all files for this split
                futures = {}
                for (pid, fname), f_seqs in pending.items():
                    future = executor.submit(
                        process_single_file, pid, fname, f_seqs, self.positive_label
                    )
                    futures[future] = (pid, fname, split)

                pbar = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Processing: {split}",
                )
                for future in pbar:
                    pid, fname, split_name = futures[future]
                    file_id = f"p1_{split_name}_{fname}"

                    results, (file_sum_v, file_sum_sq, file_count) = future.result()
                    batch = [r for r in results if r is not None]

                    if batch:
                        if not final_path.exists():
                            self._init_hdf5(final_path, batch[0]["spectrogram"].shape)
                        self._append_to_hdf5(final_path, batch)

                    # Stats already computed in worker process
                    sum_v += file_sum_v
                    sum_sq += file_sum_sq
                    count += file_count

                    self.checkpoint["completed_files"].add(file_id)
                    pbar.set_postfix({"file": fname})

                # Save per-split stats into HDF5 metadata
                if final_path.exists() and count > 0:
                    split_mean = float(sum_v / count)
                    split_std = float(np.sqrt(max(0, (sum_sq / count) - (split_mean ** 2))))
                    self.logger.info(
                        f"Split {split} stats: mean={split_mean:.6f}, std={split_std:.6f}, n={count:.0f}"
                    )
                    with h5py.File(final_path, "a") as f:
                        meta = f.require_group("metadata")
                        meta.attrs.update({
                            "sum_v": float(sum_v),
                            "sum_sq": float(sum_sq),
                            "count": float(count),
                            "norm": "none",
                        })

                self.checkpoint[f"pass1_{split}_complete"] = True
                self.save_checkpoint()

        self.logger.info(
            "Preprocessing complete (unnormalized — normalize at training time using training splits only)."
        )


if __name__ == "__main__":
    for i in range(len(PATIENTS)):
        if PATIENT_INDEX is not None and i != PATIENT_INDEX:
            continue
        cfg = get_patient_config(i)
        try:
            EEGPreprocessor(cfg).run_preprocessing()
        except Exception as e:
            print(f"Error {cfg['patient_id']}: {e}")
