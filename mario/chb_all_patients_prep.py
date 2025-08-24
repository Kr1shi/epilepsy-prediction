
#!/usr/bin/env python3
"""
CHB-MIT all-patient EEG chunker
--------------------------------
- Processes *all* patients under a CHB-MIT root directory.
- Per-patient output subfolders with chunked .npy arrays and metadata.
- Central config is provided via --config /path/config.json

Example usage:
    python chb_all_patients_prep.py --config /path/to/config.json
"""

import os
import json
import argparse
import numpy as np
import mne
import concurrent.futures
from typing import List, Dict, Optional, Tuple

# ---------- summary parser ----------
def load_seizure_summary(summary_file: str) -> Dict[str, List[Tuple[float, float]]]:
    seizure_dict: Dict[str, List[Tuple[float, float]]] = {}
    current_file = None
    seizures: List[Tuple[float, float]] = []

    if not os.path.isfile(summary_file):
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    with open(summary_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("File Name:"):
                if current_file is not None:
                    seizure_dict[current_file] = seizures
                current_file = line.split("File Name:")[1].strip()
                seizures = []
            elif line.startswith("Number of Seizures in File:"):
                num_seizures = int(line.split(":")[1].strip())
                if num_seizures == 0:
                    seizure_dict[current_file] = []
            elif line.startswith("Seizure Start Time:"):
                start_time = float(line.split(":")[1].strip().split()[0])
            elif line.startswith("Seizure End Time:"):
                end_time = float(line.split(":")[1].strip().split()[0])
                seizures.append((start_time, end_time))

        if current_file is not None and current_file not in seizure_dict:
            seizure_dict[current_file] = seizures

    return seizure_dict

# ---------- window generator ----------
def segment_and_label_iter(
    raw: mne.io.BaseRaw,
    seizure_times: List[Tuple[float, float]],
    window_size_sec: int = 30,
    stride_sec: int = 30,
    preictal_horizon_sec: int = 1800,
):
    sfreq = int(round(raw.info["sfreq"]))  # after any resample
    win_samp = window_size_sec * sfreq
    stride_samp = stride_sec * sfreq
    n_samp = raw.n_times

    for start in range(0, n_samp - win_samp + 1, stride_samp):
        end = start + win_samp

        # skip windows overlapping the actual seizure
        if any((start / sfreq < s_end) and (end / sfreq > s_start) for s_start, s_end in seizure_times):
            continue

        data, _ = raw[:, start:end]  # (n_channels, win_samp)
        # normalize per channel
        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-6)

        t_end = end / sfreq
        label = 0
        for s_start, _ in seizure_times:
            if (t_end >= s_start - preictal_horizon_sec) and (t_end <= s_start):
                label = 1
                break

        yield data, label

# ---------------- config + utils ----------------
def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = json.load(f)

    # required roots
    for k in ["chb_root_dir", "out_root_dir"]:
        if k not in cfg:
            raise ValueError(f"Missing required config key: {k}")

    # defaults
    cfg.setdefault("include_patients", [])
    cfg.setdefault("exclude_patients", [])
    cfg.setdefault("filter", {"l_freq": 0.5, "h_freq": 40.0, "fir_design": "firwin"})
    cfg.setdefault("resample_hz", 256)  # default to 256 Hz
    cfg.setdefault("window_size_sec", 30)
    cfg.setdefault("stride_sec", 30)
    cfg.setdefault("preictal_horizon_sec", 1800)
    cfg.setdefault("chunk_size", 1000)
    cfg.setdefault("standardize_channels", True)  # use per-patient intersection
    cfg.setdefault("required_channels", [])       # if provided, overrides inference
    cfg.setdefault("num_workers", 1)
    cfg.setdefault("log_level", "info")
    cfg.setdefault("on_error", "fail")          # "fail" or "skip"
    return cfg

def discover_patients(root: str, include: List[str], exclude: List[str]) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"CHB-MIT root directory not found: {root}")
    inc = {p.lower() for p in include} if include else None
    exc = {p.lower() for p in exclude} if exclude else set()
    out = []
    for name in sorted(os.listdir(root)):
        pdir = os.path.join(root, name)
        if not (os.path.isdir(pdir) and name.lower().startswith("chb")):
            continue
        if inc is not None and name.lower() not in inc:
            continue
        if name.lower() in exc:
            continue
        out.append(name)
    return out

def find_summary_file(patient_dir: str) -> str:
    cands = [f for f in os.listdir(patient_dir) if f.lower().endswith("-summary.txt")]
    if not cands:
        cands = [f for f in os.listdir(patient_dir) if f.lower().endswith("summary.txt")]
    if not cands:
        raise FileNotFoundError(f"No summary file found in {patient_dir}")
    return os.path.join(patient_dir, sorted(cands)[0])

def list_edfs(patient_dir: str) -> List[str]:
    return [os.path.join(patient_dir, f) for f in sorted(os.listdir(patient_dir)) if f.lower().endswith(".edf")]

_BAD_CH_PATTERNS = ("ecg", "ekg", "pulse", "resp", "emg", "vns", "eog")
def _is_eeg_channel(ch_name: str) -> bool:
    n = ch_name.lower()
    return not any(bad in n for bad in _BAD_CH_PATTERNS)

def infer_patient_required_channels(patient_dir: str) -> List[str]:
    """
    Header-only scan to get the intersection of EEG channels across all EDFs for this patient.
    Keeps bipolar labels as-is (e.g., 'FP1-F7'). Excludes non-EEG channels by simple name heuristics.
    """
    edfs = list_edfs(patient_dir)
    if not edfs:
        raise FileNotFoundError(f"No .edf files found in {patient_dir}")

    common = None
    for path in edfs:
        try:
            raw_hdr = mne.io.read_raw_edf(path, preload=False, verbose=False)
        except Exception as e:
            raise RuntimeError(f"Failed to read EDF header {os.path.basename(path)}: {e}") from e
        chs = [c for c in raw_hdr.ch_names if _is_eeg_channel(c)]
        if not chs:
            raise RuntimeError(f"No EEG channels detected in {os.path.basename(path)}")
        s = set(chs)
        common = s if common is None else (common & s)
        if not common:
            raise RuntimeError(f"Patient {os.path.basename(patient_dir)} has no common EEG channels across EDFs")

    return sorted(common)

# ---------- chunked preparation ----------
def prepare_dataset_chunked(
    edf_dir: str,
    summary_file: str,
    out_dir: str = "eeg_chunks",
    window_size_sec: int = 30,
    stride_sec: int = 30,
    preictal_horizon_sec: int = 1800,
    chunk_size: int = 1000,
    filter_cfg: Optional[Dict] = None,   # {"l_freq":0.5,"h_freq":40,"fir_design":"firwin"}
    resample_hz: Optional[int] = None,   # e.g., 256
    standardize_channels: bool = False,
    required_channels: Optional[List[str]] = None,
    on_error: str = "fail",             # "fail" or "skip"
):
    if not os.path.isdir(edf_dir):
        raise FileNotFoundError(f"EDF directory not found: {edf_dir}")

    seizure_dict = load_seizure_summary(summary_file)
    os.makedirs(out_dir, exist_ok=True)

    chunk_idx = 0
    buf_X: List[np.ndarray] = []
    buf_y: List[int] = []
    total_windows = 0
    total_preictal = 0
    total_interictal = 0
    processed_files = 0

    edf_files = [f for f in sorted(os.listdir(edf_dir)) if f.lower().endswith(".edf")]
    if not edf_files:
        raise FileNotFoundError(f"No .edf files found in {edf_dir}")

    # resolve required channels
    req_ch = None
    if standardize_channels:
        if required_channels:
            req_ch = list(required_channels)
        else:
            req_ch = infer_patient_required_channels(edf_dir)

    for idx, fname in enumerate(edf_files, 1):
        path = os.path.join(edf_dir, fname)
        print(f"[{idx}/{len(edf_files)}] Processing {fname}...")
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        except Exception as e:
            msg = f"  Failed to load {fname}: {e}"
            if on_error == "fail":
                raise RuntimeError(msg) from e
            print("  " + msg)
            continue

        # optional channel standardization
        if req_ch is not None:
            missing = [c for c in req_ch if c not in raw.ch_names]
            if missing:
                msg = f"  Missing required channels in {fname}: {missing[:5]}{'...' if len(missing)>5 else ''}"
                if on_error == "fail":
                    raise RuntimeError(msg)
                print(msg + " -> skipping file")
                continue
            raw.reorder_channels(req_ch)
            raw.pick_channels(req_ch, ordered=True)

        # filter
        if filter_cfg:
            l_freq = filter_cfg.get("l_freq", None)
            h_freq = filter_cfg.get("h_freq", None)
            fir_design = filter_cfg.get("fir_design", "firwin")
            raw.filter(l_freq, h_freq, fir_design=fir_design, verbose=False)

        # resample
        if resample_hz and int(round(raw.info["sfreq"])) != int(resample_hz):
            raw.resample(resample_hz, npad="auto", verbose=False)

        seizures = seizure_dict.get(fname, [])
        file_windows = 0
        file_pre = 0
        file_inter = 0

        for data, label in segment_and_label_iter(
            raw, seizures, window_size_sec, stride_sec, preictal_horizon_sec
        ):
            buf_X.append(data)
            buf_y.append(label)
            file_windows += 1
            if label == 1:
                file_pre += 1
            else:
                file_inter += 1

            if len(buf_y) >= chunk_size:
                X_chunk = np.stack(buf_X)
                y_chunk = np.array(buf_y, dtype=np.int8)
                np.save(os.path.join(out_dir, f"X_{chunk_idx}.npy"), X_chunk)
                np.save(os.path.join(out_dir, f"y_{chunk_idx}.npy"), y_chunk)
                print(f"  → Saved chunk {chunk_idx}: windows={len(y_chunk)}, preictal={int(y_chunk.sum())}")
                chunk_idx += 1
                buf_X.clear()
                buf_y.clear()

        print(f"  → file summary: windows={file_windows}, preictal={file_pre}, interictal={file_inter}")
        total_windows += file_windows
        total_preictal += file_pre
        total_interictal += file_inter
        processed_files += 1

    # final flush
    if buf_y:
        X_chunk = np.stack(buf_X)
        y_chunk = np.array(buf_y, dtype=np.int8)
        np.save(os.path.join(out_dir, f"X_{chunk_idx}.npy"), X_chunk)
        np.save(os.path.join(out_dir, f"y_{chunk_idx}.npy"), y_chunk)
        print(f"  → Saved final chunk {chunk_idx}: windows={len(y_chunk)}, preictal={int(y_chunk.sum())}")
        chunk_idx += 1

    # metadata
    meta = {
        "n_chunks": int(chunk_idx),
        "total_windows": int(total_windows),
        "preictal_windows": int(total_preictal),
        "interictal_windows": int(total_interictal),
        "channels": None if req_ch is None else list(req_ch),
        "window_length_samples": None,
        "window_size_sec": int(window_size_sec),
        "stride_sec": int(stride_sec),
        "preictal_horizon_sec": int(preictal_horizon_sec),
        "sfreq": float(raw.info["sfreq"]) if processed_files > 0 else None,
        "filter": filter_cfg or {},
        "resample_hz": resample_hz,
        "standardize_channels": bool(standardize_channels),
    }

    if chunk_idx > 0:
        sample_X = np.load(os.path.join(out_dir, "X_0.npy"), mmap_mode="r")
        if meta["channels"] is None:
            meta["channels"] = int(sample_X.shape[1])
        meta["window_length_samples"] = int(sample_X.shape[2])

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone. Summary:")
    print(f"  total windows: {total_windows}")
    print(f"  preictal: {total_preictal}, interictal: {total_interictal}")
    print(f"  chunks written: {chunk_idx}")
    print(f"  metadata saved to {os.path.join(out_dir, 'metadata.json')}")

# ---------- per-patient + orchestration ----------
def process_single_patient(patient: str, cfg: dict) -> Tuple[str, dict]:
    pdir = os.path.join(cfg["chb_root_dir"], patient)
    out_dir = os.path.join(cfg["out_root_dir"], patient)
    os.makedirs(out_dir, exist_ok=True)

    summary_file = find_summary_file(pdir)
    edfs = list_edfs(pdir)
    if not edfs:
        raise FileNotFoundError(f"{patient}: no .edf files found")

    # Decide required channels
    required_channels = cfg.get("required_channels") or None
    if cfg.get("standardize_channels", False) and not required_channels:
        required_channels = infer_patient_required_channels(pdir)
        with open(os.path.join(out_dir, "channels.json"), "w") as f:
            json.dump(required_channels, f, indent=2)

    prepare_dataset_chunked(
        edf_dir=pdir,
        summary_file=summary_file,
        out_dir=out_dir,
        window_size_sec=cfg["window_size_sec"],
        stride_sec=cfg["stride_sec"],
        preictal_horizon_sec=cfg["preictal_horizon_sec"],
        chunk_size=cfg["chunk_size"],
        filter_cfg=cfg.get("filter"),
        resample_hz=cfg.get("resample_hz"),
        standardize_channels=cfg.get("standardize_channels", False),
        required_channels=required_channels,
        on_error=cfg.get("on_error", "fail"),
    )

    # read back per-patient metadata
    meta_path = os.path.join(out_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        raise RuntimeError(f"{patient}: metadata.json not found after processing")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return patient, meta

def run_all_patients(cfg: dict):
    os.makedirs(cfg["out_root_dir"], exist_ok=True)
    patients = discover_patients(cfg["chb_root_dir"], cfg["include_patients"], cfg["exclude_patients"])
    if not patients:
        raise RuntimeError("No patient folders discovered under chb_root_dir")

    print(f"Discovered {len(patients)} patients: {', '.join(patients)}")
    results: List[Tuple[str, dict]] = []
    errors: List[Tuple[str, str]] = []

    num_workers = int(cfg.get("num_workers", 1))
    if num_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
            futs = {ex.submit(process_single_patient, p, cfg): p for p in patients}
            for fut in concurrent.futures.as_completed(futs):
                p = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    errors.append((p, str(e)))
                    if cfg.get("on_error", "fail") == "fail":
                        ex.shutdown(cancel_futures=True)
                        raise
    else:
        for p in patients:
            try:
                results.append(process_single_patient(p, cfg))
            except Exception as e:
                errors.append((p, str(e)))
                if cfg.get("on_error", "fail") == "fail":
                    raise

    # aggregate global metadata
    overall = {
        "patients": [],
        "n_chunks": 0,
        "total_windows": 0,
        "preictal_windows": 0,
        "interictal_windows": 0,
        "window_size_sec": cfg["window_size_sec"],
        "stride_sec": cfg["stride_sec"],
        "preictal_horizon_sec": cfg["preictal_horizon_sec"],
        "chunk_size": cfg["chunk_size"],
        "filter": cfg.get("filter"),
        "resample_hz": cfg.get("resample_hz"),
        "standardize_channels": cfg.get("standardize_channels", False),
        "required_channels": cfg.get("required_channels", []),
        "errors": errors,
    }

    for patient, meta in sorted(results, key=lambda x: x[0]):
        overall["patients"].append({"patient": patient, **meta})
        overall["n_chunks"] += int(meta.get("n_chunks", 0))
        overall["total_windows"] += int(meta.get("total_windows", 0))
        overall["preictal_windows"] += int(meta.get("preictal_windows", 0))
        overall["interictal_windows"] += int(meta.get("interictal_windows", 0))

    gpath = os.path.join(cfg["out_root_dir"], "global_metadata.json")
    with open(gpath, "w") as f:
        json.dump(overall, f, indent=2)

    print("\n=== All patients done ===")
    print(f"  patients processed: {len(results)}/{len(patients)}")
    print(f"  total windows: {overall['total_windows']}")
    print(f"  preictal: {overall['preictal_windows']}, interictal: {overall['interictal_windows']}")
    print(f"  total chunks: {overall['n_chunks']}")
    print(f"  global metadata saved to {gpath}")
    if errors and cfg.get("on_error", "fail") != "fail":
        print("  some errors (skipped):")
        for p, msg in errors:
            print(f"   - {p}: {msg}")

# ---------- CLI entrypoint ----------
def parse_cli():
    p = argparse.ArgumentParser(description="All-patient chunked EEG preprocessing for CHB-MIT")
    p.add_argument("--config", required=True, help="Path to JSON config")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_cli()
    cfg = load_config(args.config)
    run_all_patients(cfg)
