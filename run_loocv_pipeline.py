#!/usr/bin/env python3
"""
Utility script to run the full LOOCV pipeline (segmentation → preprocessing →
training → evaluation) for multiple seizures/folds and report the mean accuracy.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean

from data_segmentation_helpers import config as cfg

REPO_ROOT = Path(__file__).resolve().parent

STAGES = [
    ("Segmentation", ["data_segmentation.py"]),
    ("Preprocessing", ["data_preprocessing.py"]),
    ("Training", ["train.py"]),
]

EVAL_STAGE = ("Evaluation", ["evaluate_test.py"])
ACCURACY_PATTERN = re.compile(r"Accuracy:\s+([0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LOOCV pipeline across folds and average accuracy.")
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        help="Explicit fold indices to run (defaults to all folds defined in config).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N folds (ignored when --folds is provided).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for each stage (default: current interpreter).",
    )
    return parser.parse_args()


def run_stage(stage_name: str, python_bin: str, script: list[str], env: dict[str, str]) -> None:
    cmd = [python_bin, *script]
    print(f"\n[{stage_name}] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def run_evaluation(python_bin: str, script: list[str], env: dict[str, str]) -> float:
    cmd = [python_bin, *script]
    print(f"\n[{EVAL_STAGE[0]}] Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    # Echo evaluation output for visibility
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    match = ACCURACY_PATTERN.search(result.stdout)
    if not match:
        raise RuntimeError("Failed to parse accuracy from evaluate_test.py output.")
    return float(match.group(1))


def build_fold_list(args: argparse.Namespace) -> list[int]:
    if args.folds:
        return args.folds

    max_folds = cfg.LOOCV_TOTAL_SEIZURES
    folds = list(range(max_folds))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        folds = folds[: args.limit]
    return folds


def main() -> None:
    args = parse_args()
    folds = build_fold_list(args)

    if not folds:
        print("No folds specified. Nothing to do.")
        return

    accuracy_by_fold = []

    for fold in folds:
        if fold < 0 or fold >= cfg.LOOCV_TOTAL_SEIZURES:
            raise ValueError(f"Fold {fold} is outside 0..{cfg.LOOCV_TOTAL_SEIZURES - 1}")

        print("\n" + "=" * 80)
        print(f"Starting fold {fold} / {cfg.LOOCV_TOTAL_SEIZURES - 1}")
        print("=" * 80)

        env = os.environ.copy()
        env["LOOCV_FOLD_ID"] = str(fold)

        # Keep metadata aligned with the patient-specific total if it was overridden externally.
        env["LOOCV_TOTAL_SEIZURES"] = str(cfg.LOOCV_TOTAL_SEIZURES)
        env["SINGLE_PATIENT_ID"] = cfg.SINGLE_PATIENT_ID

        for stage_name, script in STAGES:
            run_stage(stage_name, args.python, script, env)

        accuracy = run_evaluation(args.python, EVAL_STAGE[1], env)
        accuracy_by_fold.append((fold, accuracy))
        print(f"[Fold {fold}] Accuracy: {accuracy * 100:.2f}%")

    print("\n" + "=" * 80)
    print("LOOCV SUMMARY")
    print("=" * 80)
    for fold, acc in accuracy_by_fold:
        print(f"Fold {fold:02d}: {acc * 100:.2f}%")

    avg_accuracy = mean(acc for _, acc in accuracy_by_fold)
    print(f"\nAverage accuracy ({len(accuracy_by_fold)} folds): {avg_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
