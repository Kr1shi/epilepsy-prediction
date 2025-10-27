"""Visualization script for CNN-LSTM sequence-based preprocessed data

This script validates preprocessed HDF5 data by visualizing:
1. Individual sequence spectrograms (temporal progression across segments)
2. Class distribution and balance
3. Spectrogram shape validation
4. Sample positive class (preictal/ictal) vs interictal sequences

Supports both task modes:
- Prediction: preictal vs interictal
- Detection: ictal vs interictal
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
import logging
from data_segmentation_helpers.config import *

def get_positive_label():
    """Get positive class label based on task mode"""
    return 'preictal' if TASK_MODE == 'prediction' else 'ictal'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SequenceVisualizer:
    """Visualize preprocessed CNN-LSTM sequence data from HDF5 files"""

    def __init__(self, data_dir=None):
        if data_dir is None:
            self.data_dir = Path("preprocessing") / "data" / OUTPUT_PREFIX
        else:
            self.data_dir = Path(data_dir)
        self.output_dir = Path('preprocessing/visualizations') / OUTPUT_PREFIX
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("="*60)
        logger.info("CNN-LSTM SEQUENCE VISUALIZATION")
        logger.info("="*60)
        logger.info(f"Task mode: {TASK_MODE.upper()} ({get_positive_label()} vs interictal)")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Output prefix: {OUTPUT_PREFIX}")

    def load_dataset_info(self, split='train'):
        """Load dataset metadata and sample data"""
        h5_file = self.data_dir / f"{split}_dataset.h5"

        if not h5_file.exists():
            raise FileNotFoundError(f"Dataset not found: {h5_file}")

        with h5py.File(h5_file, 'r') as f:
            # Get shapes and metadata
            info = {
                'n_sequences': f['spectrograms'].shape[0],
                'sequence_length': f['spectrograms'].shape[1],
                'n_channels': f['spectrograms'].shape[2],
                'n_frequencies': f['spectrograms'].shape[3],
                'n_time_bins': f['spectrograms'].shape[4],
                'labels': f['labels'][:],
                'patient_ids': [pid.decode('utf-8') for pid in f['patient_ids'][:]],
            }

            # Get metadata if available
            if 'metadata' in f:
                metadata = dict(f['metadata'].attrs)
                # Handle both bytes and strings for channel names
                channels_raw = metadata.get('target_channels', [])
                info['target_channels'] = [
                    ch.decode('utf-8') if isinstance(ch, bytes) else ch
                    for ch in channels_raw
                ]
                info['frequencies'] = metadata.get('frequencies', [])
                info['frequency_range'] = metadata.get('frequency_range', [])
                info['sequence_length_config'] = metadata.get('sequence_length', SEQUENCE_LENGTH)
                info['segment_duration'] = metadata.get('segment_duration', SEGMENT_DURATION)

        positive_label = get_positive_label()

        logger.info(f"\n{split.upper()} Dataset Info:")
        logger.info(f"  Task mode: {TASK_MODE.upper()} ({positive_label} vs interictal)")
        logger.info(f"  Total sequences: {info['n_sequences']}")
        logger.info(f"  Sequence shape: ({info['sequence_length']}, {info['n_channels']}, {info['n_frequencies']}, {info['n_time_bins']})")
        logger.info(f"  Sequence length: {info['sequence_length']} segments")
        logger.info(f"  Channels: {info['n_channels']}")
        logger.info(f"  Frequencies: {info['n_frequencies']} bins")
        logger.info(f"  Time bins per segment: {info['n_time_bins']}")
        logger.info(f"  Class 0 (interictal): {np.sum(info['labels'] == 0)}")
        logger.info(f"  Class 1 ({positive_label}): {np.sum(info['labels'] == 1)}")

        return info

    def plot_class_distribution(self, splits=['train', 'val', 'test']):
        """Plot class distribution across all splits"""
        positive_label = get_positive_label()

        fig, axes = plt.subplots(1, len(splits), figsize=(5*len(splits), 4))
        if len(splits) == 1:
            axes = [axes]

        fig.suptitle(f'Class Distribution Across Splits\n({TASK_MODE.upper()}: {positive_label.capitalize()} vs Interictal)',
                    fontsize=16, fontweight='bold')

        for idx, split in enumerate(splits):
            h5_file = self.data_dir / f"{split}_dataset.h5"
            if not h5_file.exists():
                logger.warning(f"Skipping {split} - file not found")
                continue

            with h5py.File(h5_file, 'r') as f:
                labels = f['labels'][:]

            # Count classes
            unique, counts = np.unique(labels, return_counts=True)

            # Plot
            ax = axes[idx]
            bars = ax.bar(['Interictal (0)', f'{positive_label.capitalize()} (1)'], counts, color=['#2ecc71', '#e74c3c'])
            ax.set_ylabel('Number of Sequences')
            ax.set_title(f'{split.upper()} Split')
            ax.set_ylim(0, max(counts) * 1.2)

            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(count)}\n({count/sum(counts)*100:.1f}%)',
                       ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        output_file = self.output_dir / 'class_distribution.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Saved class distribution plot to {output_file}")
        plt.close()

    def plot_sequence_spectrograms(self, split='train', n_examples=2):
        """Plot sequences from same patients (optimized with smart sampling)

        Compares positive class (ictal/preictal) vs interictal from the SAME patient
        for better visual comparison of seizure vs non-seizure activity.
        """
        positive_label = get_positive_label()
        h5_file = self.data_dir / f"{split}_dataset.h5"

        with h5py.File(h5_file, 'r') as f:
            # Efficient indexing: only load labels to find indices
            labels = f['labels'][:]
            patient_ids_raw = [pid.decode('utf-8') for pid in f['patient_ids'][:]]
            spectrograms = f['spectrograms']

            # Get metadata
            if 'metadata' in f:
                metadata = dict(f['metadata'].attrs)
                # Handle both bytes and strings for channel names
                channels_raw = metadata.get('target_channels', [])
                channels = [ch.decode('utf-8') if isinstance(ch, bytes) else ch for ch in channels_raw]
            else:
                channels = [f'Ch{i}' for i in range(spectrograms.shape[2])]

            # Find patients with BOTH positive and interictal sequences
            positive_indices = np.where(labels == 1)[0]
            interictal_indices = np.where(labels == 0)[0]

            # Group indices by patient
            patient_positive = {}  # patient_id -> list of positive indices
            patient_interictal = {}  # patient_id -> list of interictal indices

            for idx in positive_indices:
                pid = patient_ids_raw[idx]
                if pid not in patient_positive:
                    patient_positive[pid] = []
                patient_positive[pid].append(idx)

            for idx in interictal_indices:
                pid = patient_ids_raw[idx]
                if pid not in patient_interictal:
                    patient_interictal[pid] = []
                patient_interictal[pid].append(idx)

            # Find patients with both classes
            patients_with_both = set(patient_positive.keys()) & set(patient_interictal.keys())

            if not patients_with_both:
                logger.warning(f"No patients found with both {positive_label} and interictal sequences!")
                return

            # Sample patients
            selected_patients = random.sample(list(patients_with_both), min(n_examples, len(patients_with_both)))

            logger.info(f"\nPlotting {len(selected_patients)} patients with both {positive_label} and interictal sequences")

            # For each patient, plot one positive and one interictal sequence
            for patient_id in selected_patients:
                # Sample one sequence from each class for this patient
                pos_idx = random.choice(patient_positive[patient_id])
                int_idx = random.choice(patient_interictal[patient_id])

                logger.info(f"  Patient {patient_id}: {positive_label} seq #{pos_idx}, interictal seq #{int_idx}")

                # Plot positive sequence
                sequence = spectrograms[pos_idx]
                self._plot_single_sequence(sequence, patient_id, positive_label, pos_idx, channels)

                # Plot interictal sequence
                sequence = spectrograms[int_idx]
                self._plot_single_sequence(sequence, patient_id, 'interictal', int_idx, channels)

    def _plot_single_sequence(self, sequence, patient_id, label, seq_idx, channels):
        """Plot a single sequence (selected segments and key channels for performance)"""
        seq_len, n_channels, n_freqs, n_time_bins = sequence.shape

        # Select 6 representative segments: first 2, middle 2, last 2
        segment_indices = [
            0, 1,                              # Beginning
            seq_len // 2 - 1, seq_len // 2,    # Middle
            seq_len - 2, seq_len - 1           # End
        ]
        segment_labels = [
            f'Seg 1\n(0-{SEGMENT_DURATION}s)',
            f'Seg 2\n({SEGMENT_DURATION}-{SEGMENT_DURATION*2}s)',
            f'Seg {seq_len//2}\n({(seq_len//2-1)*SEGMENT_DURATION}-{(seq_len//2)*SEGMENT_DURATION}s)',
            f'Seg {seq_len//2+1}\n({(seq_len//2)*SEGMENT_DURATION}-{(seq_len//2+1)*SEGMENT_DURATION}s)',
            f'Seg {seq_len-1}\n({(seq_len-2)*SEGMENT_DURATION}-{(seq_len-1)*SEGMENT_DURATION}s)',
            f'Seg {seq_len}\n({(seq_len-1)*SEGMENT_DURATION}-{seq_len*SEGMENT_DURATION}s)'
        ]

        # Select 6 key channels representing different brain regions
        KEY_CHANNELS = ['FP1-F7', 'FP2-F8', 'C3-P3', 'C4-P4', 'T7-P7', 'T8-P8']
        channel_indices = [channels.index(ch) for ch in KEY_CHANNELS if ch in channels]

        # Fallback: if key channels not found, use evenly spaced channels
        if len(channel_indices) < 6:
            step = n_channels // 6
            channel_indices = list(range(0, n_channels, step))[:6]

        # Create optimized figure: 6 channels × 6 segments = 36 subplots (vs 540 before)
        fig, axes = plt.subplots(len(channel_indices), len(segment_indices),
                                figsize=(len(segment_indices) * 2.5, len(channel_indices) * 1.5),
                                sharex=True, sharey=True)

        fig.suptitle(f'Sequence #{seq_idx} - {patient_id} - {label.upper()}\n'
                    f'Showing 6 of {seq_len} segments (beginning, middle, end) × 6 key channels\n'
                    f'Total sequence: {seq_len * SEGMENT_DURATION}s',
                    fontsize=14, fontweight='bold')

        # Plot each selected channel × segment combination
        for ch_plot_idx, ch_idx in enumerate(channel_indices):
            for seg_plot_idx, seg_idx in enumerate(segment_indices):
                ax = axes[ch_plot_idx, seg_plot_idx] if len(channel_indices) > 1 else axes[seg_plot_idx]

                # Get spectrogram for this segment and channel
                spec = sequence[seg_idx, ch_idx, :, :]  # (n_freqs, n_time_bins)

                # Plot
                im = ax.imshow(spec, aspect='auto', origin='lower',
                             cmap='viridis', interpolation='nearest')

                # Labels
                if seg_plot_idx == 0:
                    ax.set_ylabel(f'{channels[ch_idx] if ch_idx < len(channels) else f"Ch{ch_idx}"}\nFreq',
                                fontsize=8)
                if ch_plot_idx == 0:
                    ax.set_title(segment_labels[seg_plot_idx], fontsize=8)
                if ch_plot_idx == len(channel_indices) - 1:
                    ax.set_xlabel('Time', fontsize=7)

                # Remove ticks for cleaner look
                ax.set_xticks([])
                ax.set_yticks([])

        # Add colorbar
        fig.colorbar(im, ax=axes.ravel().tolist() if len(channel_indices) > 1 else axes,
                    label='Log Power', shrink=0.8)

        plt.tight_layout()

        output_file = self.output_dir / f'sequence_{label}_{patient_id}_{seq_idx}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Saved sequence plot to {output_file}")
        plt.close()

    def plot_channel_comparison(self, split='train', n_examples=1):
        """Plot all channels side-by-side for a single segment (same patient comparison)"""
        positive_label = get_positive_label()
        h5_file = self.data_dir / f"{split}_dataset.h5"

        with h5py.File(h5_file, 'r') as f:
            labels = f['labels'][:]
            spectrograms = f['spectrograms']
            patient_ids_raw = [pid.decode('utf-8') for pid in f['patient_ids'][:]]

            # Get metadata
            if 'metadata' in f:
                metadata = dict(f['metadata'].attrs)
                # Handle both bytes and strings for channel names
                channels_raw = metadata.get('target_channels', [])
                channels = [ch.decode('utf-8') if isinstance(ch, bytes) else ch for ch in channels_raw]
                frequencies = metadata.get('frequencies', [])
            else:
                channels = [f'Ch{i}' for i in range(spectrograms.shape[2])]
                frequencies = None

            # Find patients with BOTH classes
            positive_indices = np.where(labels == 1)[0]
            interictal_indices = np.where(labels == 0)[0]

            # Group by patient
            patient_positive = {}
            patient_interictal = {}

            for idx in positive_indices:
                pid = patient_ids_raw[idx]
                if pid not in patient_positive:
                    patient_positive[pid] = []
                patient_positive[pid].append(idx)

            for idx in interictal_indices:
                pid = patient_ids_raw[idx]
                if pid not in patient_interictal:
                    patient_interictal[pid] = []
                patient_interictal[pid].append(idx)

            # Find patients with both
            patients_with_both = set(patient_positive.keys()) & set(patient_interictal.keys())

            if not patients_with_both:
                logger.warning(f"No patients with both classes for channel comparison")
                return

            # Pick one patient
            patient_id = random.choice(list(patients_with_both))
            positive_idx = random.choice(patient_positive[patient_id])
            interictal_idx = random.choice(patient_interictal[patient_id])

            logger.info(f"Channel comparison for patient {patient_id}: {positive_label} seq #{positive_idx}, interictal seq #{interictal_idx}")

            for idx, label in [(positive_idx, positive_label), (interictal_idx, 'interictal')]:
                sequence = spectrograms[idx]  # (seq_len, n_channels, n_freqs, n_time_bins)

                # Plot first segment, all channels
                first_segment = sequence[0]  # (n_channels, n_freqs, n_time_bins)

                n_channels = first_segment.shape[0]
                n_cols = 3
                n_rows = (n_channels + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
                axes = axes.flatten() if n_channels > 1 else [axes]

                fig.suptitle(f'All Channels - {patient_id} - {label.upper()} (First Segment)',
                           fontsize=14, fontweight='bold')

                for ch_idx in range(n_channels):
                    ax = axes[ch_idx]
                    spec = first_segment[ch_idx, :, :]

                    extent = [0, SEGMENT_DURATION,
                            frequencies[0] if frequencies is not None else 0,
                            frequencies[-1] if frequencies is not None else spec.shape[0]]

                    im = ax.imshow(spec, aspect='auto', origin='lower',
                                 extent=extent, cmap='viridis')
                    ax.set_title(channels[ch_idx] if ch_idx < len(channels) else f'Channel {ch_idx}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Frequency (Hz)')
                    plt.colorbar(im, ax=ax, label='Log Power')

                # Hide extra subplots
                for ax_idx in range(n_channels, len(axes)):
                    axes[ax_idx].axis('off')

                plt.tight_layout()

                output_file = self.output_dir / f'channels_comparison_{label}_{patient_id}.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                logger.info(f"Saved channel comparison to {output_file}")
                plt.close()

    def validate_data_shapes(self, splits=['train', 'val', 'test']):
        """Validate that data shapes match model expectations"""
        logger.info("\n" + "="*60)
        logger.info("DATA SHAPE VALIDATION")
        logger.info("="*60)

        expected_seq_len = SEQUENCE_LENGTH
        expected_channels = len(TARGET_CHANNELS)

        issues = []

        for split in splits:
            h5_file = self.data_dir / f"{split}_dataset.h5"
            if not h5_file.exists():
                logger.warning(f"Skipping {split} - file not found")
                continue

            with h5py.File(h5_file, 'r') as f:
                spec_shape = f['spectrograms'].shape
                label_shape = f['labels'].shape

                logger.info(f"\n{split.upper()} Split:")
                logger.info(f"  Spectrograms shape: {spec_shape}")
                logger.info(f"  Labels shape: {label_shape}")
                logger.info(f"  Expected: (N, {expected_seq_len}, {expected_channels}, freq, time)")

                # Validate
                if spec_shape[1] != expected_seq_len:
                    issue = f"{split}: Sequence length mismatch - got {spec_shape[1]}, expected {expected_seq_len}"
                    issues.append(issue)
                    logger.error(f"  ❌ {issue}")
                else:
                    logger.info(f"  ✓ Sequence length correct: {spec_shape[1]}")

                if spec_shape[2] != expected_channels:
                    issue = f"{split}: Channel count mismatch - got {spec_shape[2]}, expected {expected_channels}"
                    issues.append(issue)
                    logger.error(f"  ❌ {issue}")
                else:
                    logger.info(f"  ✓ Channel count correct: {spec_shape[2]}")

                if spec_shape[0] != label_shape[0]:
                    issue = f"{split}: Mismatch between spectrograms and labels - {spec_shape[0]} vs {label_shape[0]}"
                    issues.append(issue)
                    logger.error(f"  ❌ {issue}")
                else:
                    logger.info(f"  ✓ Spectrograms and labels aligned: {spec_shape[0]} sequences")

                # Check for NaN or Inf
                sample = f['spectrograms'][0]
                if np.any(np.isnan(sample)):
                    issue = f"{split}: Found NaN values in spectrograms!"
                    issues.append(issue)
                    logger.error(f"  ❌ {issue}")
                else:
                    logger.info(f"  ✓ No NaN values detected")

                if np.any(np.isinf(sample)):
                    issue = f"{split}: Found Inf values in spectrograms!"
                    issues.append(issue)
                    logger.error(f"  ❌ {issue}")
                else:
                    logger.info(f"  ✓ No Inf values detected")

                # Value range check
                min_val = np.min(sample)
                max_val = np.max(sample)
                logger.info(f"  Value range: [{min_val:.4f}, {max_val:.4f}]")

        if issues:
            logger.error("\n" + "="*60)
            logger.error(f"VALIDATION FAILED - {len(issues)} issues found:")
            for issue in issues:
                logger.error(f"  - {issue}")
            logger.error("="*60)
        else:
            logger.info("\n" + "="*60)
            logger.info("✅ ALL VALIDATIONS PASSED")
            logger.info("="*60)

        return len(issues) == 0

    def run_full_validation(self, splits=['train', 'val', 'test'], n_examples=2):
        """Run complete validation and visualization suite"""
        logger.info("Starting full validation and visualization...")

        # 1. Validate shapes
        shapes_valid = self.validate_data_shapes(splits)

        # 2. Load and display info
        for split in splits:
            try:
                self.load_dataset_info(split)
            except FileNotFoundError as e:
                logger.warning(f"Skipping {split}: {e}")

        # 3. Plot class distribution
        logger.info("\nGenerating class distribution plots...")
        self.plot_class_distribution(splits)

        # 4. Plot example sequences
        logger.info(f"\nGenerating {n_examples} example sequence plots per class...")
        self.plot_sequence_spectrograms('train', n_examples)

        # 5. Plot channel comparisons
        logger.info("\nGenerating channel comparison plots...")
        self.plot_channel_comparison('train', n_examples=1)

        logger.info("\n" + "="*60)
        logger.info("VISUALIZATION COMPLETE")
        logger.info(f"All plots saved to: {self.output_dir}")
        logger.info("="*60)

        return shapes_valid


if __name__ == "__main__":
    visualizer = SequenceVisualizer()

    # Run full validation
    valid = visualizer.run_full_validation(
        splits=['train', 'val', 'test'],
        n_examples=2
    )

    if not valid:
        logger.error("\n⚠️  DATA VALIDATION FAILED - Check logs above for details")
        logger.error("⚠️  Model will not train correctly with invalid data!")
    else:
        logger.info("\n✅ Data validation passed - Ready for training")
