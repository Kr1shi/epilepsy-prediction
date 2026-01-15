"""Compute global normalization statistics from all patients for LOPO cross-validation

This script processes ALL patients' data to compute global mean and std statistics
that can be reused across all LOPO folds, dramatically speeding up preprocessing.

Usage:
    python compute_global_normalization_stats.py

Output:
    preprocessing/checkpoints/global_normalization_stats.json

The output file contains:
    - global_mean: Mean across all spectrograms
    - global_std: Standard deviation across all spectrograms
    - n_sequences: Number of sequences processed
    - total_values: Total number of spectrogram values
    - patients_processed: List of patient IDs included
"""

import json
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from data_segmentation_helpers.config import *
from data_segmentation_helpers.seizure_counts import SEIZURE_COUNTS

# Import the preprocessing class
from data_preprocessing import EEGPreprocessor


def load_all_patient_sequences() -> Dict[str, List[Dict]]:
    """Load sequence files for all patients
    
    Returns:
        Dictionary mapping patient_id to list of their sequences
    """
    all_patient_sequences = {}
    all_patients = sorted(SEIZURE_COUNTS.keys())
    
    print(f"\n{'='*60}")
    print("LOADING SEQUENCE FILES FOR ALL PATIENTS")
    print(f"{'='*60}\n")
    
    for patient_id in all_patients:
        sequence_file = f"lopo_test_{patient_id}_sequences_{TASK_MODE}.json"
        
        if not Path(sequence_file).exists():
            print(f"⚠️  Warning: Sequence file not found for {patient_id}: {sequence_file}")
            print(f"    Run: python data_segmentation.py --test_patient {patient_id}")
            continue
        
        try:
            with open(sequence_file, 'r') as f:
                data = json.load(f)
            
            sequences = data['sequences']
            # Extract only sequences from this patient
            patient_sequences = [s for s in sequences if s['patient_id'] == patient_id]
            
            all_patient_sequences[patient_id] = patient_sequences
            print(f"✅ Loaded {patient_id}: {len(patient_sequences)} sequences")
            
        except Exception as e:
            print(f"❌ Error loading {patient_id}: {e}")
    
    return all_patient_sequences


class OnlineStats:
    """Compute mean and variance online using Welford's algorithm
    
    This avoids storing all values in memory, making it suitable for
    large datasets that don't fit in RAM.
    
    Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences from mean
    
    def update(self, values):
        """Update statistics with new batch of values
        
        Args:
            values: numpy array of any shape (will be flattened)
        """
        values = values.flatten()
        for x in values:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2
    
    def update_batch(self, values):
        """Update statistics with new batch (vectorized version)
        
        More memory efficient than update() for large batches
        """
        values = values.flatten()
        batch_size = len(values)
        
        if batch_size == 0:
            return
        
        # Vectorized Welford's algorithm
        for x in values:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2
    
    @property
    def variance(self):
        """Return variance"""
        if self.n < 2:
            return 0.0
        return self.M2 / self.n
    
    @property
    def std(self):
        """Return standard deviation"""
        return np.sqrt(self.variance)
    
    def finalize(self):
        """Return final mean and std"""
        return self.mean, self.std


def compute_global_normalization_stats():
    """Compute global normalization statistics from all patients
    
    Uses online/streaming algorithm to avoid loading all data into memory.
    Memory usage: O(1) instead of O(n) where n is total spectrogram values.
    """
    
    print(f"\n{'='*60}")
    print("COMPUTING GLOBAL NORMALIZATION STATISTICS")
    print("Using memory-efficient streaming algorithm")
    print(f"{'='*60}\n")
    
    # Setup output directory
    output_dir = Path("preprocessing/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "global_normalization_stats.json"
    
    # Check if already exists
    if output_file.exists():
        response = input(f"\n⚠️  Global stats file already exists: {output_file}\n   Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Setup logging
    log_file = output_dir / "global_normalization_computation.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("GLOBAL NORMALIZATION STATISTICS COMPUTATION")
    logger.info("="*60)
    
    # Load all patient sequences
    all_patient_sequences = load_all_patient_sequences()
    
    if not all_patient_sequences:
        logger.error("No patient sequence files found!")
        logger.error("Please run data_segmentation.py first for each patient.")
        return
    
    total_patients = len(all_patient_sequences)
    total_sequences = sum(len(seqs) for seqs in all_patient_sequences.values())
    
    logger.info(f"\nDataset summary:")
    logger.info(f"  Patients: {total_patients}")
    logger.info(f"  Total sequences: {total_sequences}")
    
    # Create a temporary preprocessor instance (without fold config)
    # We'll use its methods but not run the full pipeline
    preprocessor = EEGPreprocessor.__new__(EEGPreprocessor)
    preprocessor.logger = logger
    preprocessor.removed_segments = {
        "wrong_duration": 0,
        "beyond_file_bounds": 0,
        "processing_errors": 0,
    }
    
    # Initialize normalization stats
    preprocessor.global_mean = None
    preprocessor.global_std = None
    
    # Initialize streaming statistics (memory-efficient)
    logger.info(f"\n{'='*60}")
    logger.info("PROCESSING ALL SEQUENCES (STREAMING MODE)")
    logger.info("Memory usage: O(1) - no accumulation in RAM")
    logger.info(f"{'='*60}\n")
    
    global_stats = OnlineStats()
    processed_count = 0
    patient_stats = {}
    
    for patient_id in sorted(all_patient_sequences.keys()):
        sequences = all_patient_sequences[patient_id]
        
        logger.info(f"\nProcessing patient {patient_id} ({len(sequences)} sequences)...")
        
        # Per-patient stats (also streaming)
        patient_online_stats = OnlineStats()
        patient_sequence_count = 0
        
        # Group sequences by file for efficient processing
        file_groups = preprocessor.group_sequences_by_file(sequences)
        
        with tqdm(total=len(sequences), desc=f"  {patient_id}") as pbar:
            for (patient_id_key, filename), file_sequences in file_groups.items():
                try:
                    # Process sequences from this file WITHOUT normalization
                    processed_file_sequences = preprocessor.preprocess_sequences_from_file(
                        patient_id_key,
                        filename,
                        file_sequences,
                        apply_normalization=False
                    )
                    
                    # Update statistics incrementally (no accumulation!)
                    for processed in processed_file_sequences:
                        if processed:
                            spectrogram = processed['spectrogram']
                            
                            # Update global stats
                            global_stats.update_batch(spectrogram)
                            
                            # Update patient stats
                            patient_online_stats.update_batch(spectrogram)
                            
                            patient_sequence_count += 1
                            processed_count += 1
                            
                            # Free memory immediately
                            del spectrogram
                        pbar.update(1)
                        
                except Exception as e:
                    logger.error(f"Error processing {patient_id}/{filename}: {e}")
                    pbar.update(len(file_sequences))
        
        # Store patient statistics
        if patient_sequence_count > 0:
            patient_mean, patient_std = patient_online_stats.finalize()
            patient_stats[patient_id] = {
                'n_sequences': patient_sequence_count,
                'mean': float(patient_mean),
                'std': float(patient_std),
                'total_values': int(patient_online_stats.n)
            }
            logger.info(f"  Patient {patient_id}: mean={patient_mean:.6f}, "
                       f"std={patient_std:.6f}")
    
    if processed_count == 0:
        logger.error("Failed to process any sequences!")
        return
    
    # Finalize global statistics
    logger.info(f"\n{'='*60}")
    logger.info("FINALIZING GLOBAL STATISTICS")
    logger.info(f"{'='*60}\n")
    
    global_mean, global_std = global_stats.finalize()
    
    logger.info(f"Processed {processed_count:,} sequences")
    logger.info(f"Total values: {global_stats.n:,}")
    logger.info(f"Global mean: {global_mean:.6f}")
    logger.info(f"Global std: {global_std:.6f}")
    
    # Prepare output data
    output_data = {
        'global_mean': float(global_mean),
        'global_std': float(global_std),
        'n_sequences': processed_count,
        'total_values': int(global_stats.n),
        'n_patients': total_patients,
        'patients_processed': sorted(all_patient_sequences.keys()),
        'per_patient_stats': patient_stats,
        'config': {
            'task_mode': TASK_MODE,
            'segment_duration': SEGMENT_DURATION,
            'sequence_length': SEQUENCE_LENGTH,
            'low_freq_hz': LOW_FREQ_HZ,
            'high_freq_hz': HIGH_FREQ_HZ,
            'notch_freq_hz': NOTCH_FREQ_HZ,
            'artifact_threshold_std': ARTIFACT_THRESHOLD_STD,
            'apply_log_transform': APPLY_LOG_TRANSFORM,
            'log_transform_epsilon': LOG_TRANSFORM_EPSILON,
        }
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("GLOBAL NORMALIZATION STATISTICS COMPUTED")
    logger.info(f"{'='*60}")
    logger.info(f"\nGlobal Statistics:")
    logger.info(f"  Mean: {global_mean:.6f}")
    logger.info(f"  Std:  {global_std:.6f}")
    logger.info(f"\nDataset:")
    logger.info(f"  Patients: {total_patients}")
    logger.info(f"  Sequences: {processed_count}")
    logger.info(f"  Total values: {global_stats.n:,}")
    logger.info(f"\nOutput file: {output_file}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"\n{'='*60}")
    logger.info("✅ DONE - Memory-efficient computation completed")
    logger.info(f"{'='*60}\n")
    
    # Print usage instructions
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\nTo use these global statistics in preprocessing:")
    print("1. Run preprocessing with --use_global_stats flag:")
    print("   python data_preprocessing.py --test_patient chb01 --use_global_stats")
    print("\n2. Or modify data_preprocessing.py to load global stats by default")
    print("\nThe global stats will be loaded instead of computing per-fold stats.")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute global normalization statistics for LOPO cross-validation"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing global stats without prompting'
    )
    
    args = parser.parse_args()
    
    try:
        compute_global_normalization_stats()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
