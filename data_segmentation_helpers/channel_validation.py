"""Channel validation utilities for EEG data segmentation

This module validates that EEG files contain all required channels before
creating segments/sequences. This allows us to fail fast and avoid wasting
compute on data that cannot be processed.
"""

import mne
import warnings
from typing import Tuple, List, Dict, Set
from .config import TARGET_CHANNELS, SKIP_CHANNEL_VALIDATION

# Suppress MNE warnings during validation
warnings.filterwarnings("ignore", message="Channel names are not unique")
warnings.filterwarnings("ignore", message=".*duplicates.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Scaling factor is not defined")
warnings.filterwarnings("ignore", message=".*scaling factor.*", category=RuntimeWarning)

# Set MNE log level to ERROR to reduce noise
mne.set_log_level('ERROR')


def get_channel_mapping(raw) -> Dict[str, str]:
    """Get mapping from target channels to actual channel names in file

    Handles duplicate channels that MNE renames (e.g., 'FP1-F7' -> 'FP1-F7-0')

    Args:
        raw: MNE Raw object

    Returns:
        Dictionary mapping target channel names to actual names in file
        Example: {'FP1-F7': 'FP1-F7-0', 'F7-T7': 'F7-T7', ...}
    """
    available_channels = set(raw.ch_names)
    channel_mapping = {}

    for target_ch in TARGET_CHANNELS:
        if target_ch in available_channels:
            # Exact match found
            channel_mapping[target_ch] = target_ch
        else:
            # Check for renamed duplicates (e.g., 'FP1-F7-0', 'FP1-F7-1')
            duplicate_matches = [
                ch for ch in available_channels
                if ch.startswith(target_ch + '-') and ch.split('-')[-1].isdigit()
            ]
            if duplicate_matches:
                # Use first duplicate match
                channel_mapping[target_ch] = duplicate_matches[0]

    return channel_mapping


def validate_file_channels(edf_path: str, target_channels: List[str] = None) -> Tuple[bool, Set[str], List[str], Dict[str, str]]:
    """Validate that an EDF file contains all required channels

    Args:
        edf_path: Path to EDF file
        target_channels: List of required channel names (defaults to TARGET_CHANNELS from config)

    Returns:
        Tuple of:
        - is_valid (bool): True if all target channels are available
        - available_channels (set): All channels found in the file
        - missing_channels (list): Channels from target_channels that are missing
        - channel_mapping (dict): Mapping from target to actual channel names
    """
    if target_channels is None:
        target_channels = TARGET_CHANNELS

    # If validation is disabled, assume all files are valid
    if SKIP_CHANNEL_VALIDATION:
        return True, set(), [], {}

    try:
        # Read EDF header only (fast - doesn't load data)
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        available_channels = set(raw.ch_names)

        # Get channel mapping (handles duplicates)
        channel_mapping = get_channel_mapping(raw)

        # Find missing channels
        missing_channels = []
        for target_ch in target_channels:
            if target_ch not in channel_mapping:
                missing_channels.append(target_ch)

        is_valid = len(missing_channels) == 0

        return is_valid, available_channels, missing_channels, channel_mapping

    except Exception as e:
        # If we can't read the file, consider it invalid
        return False, set(), target_channels, {}


def validate_patient_files(patient_id: str, file_list: List[str], base_path: str) -> Dict:
    """Validate all files for a patient

    Args:
        patient_id: Patient identifier (e.g., 'chb01')
        file_list: List of EDF filenames for this patient
        base_path: Base path to dataset (e.g., 'physionet.org/files/chbmit/1.0.0/')

    Returns:
        Dictionary with validation results:
        {
            'patient_id': str,
            'total_files': int,
            'valid_files': int,
            'invalid_files': int,
            'valid_file_list': List[str],
            'invalid_file_info': List[dict],  # [{file, missing_channels}, ...]
            'all_missing_channels': Set[str]  # All unique missing channels across files
        }
    """
    valid_files = []
    invalid_file_info = []
    all_missing_channels = set()

    for filename in file_list:
        edf_path = f"{base_path}{patient_id}/{filename}"
        is_valid, available_ch, missing_ch, mapping = validate_file_channels(edf_path)

        if is_valid:
            valid_files.append(filename)
        else:
            invalid_file_info.append({
                'file': filename,
                'missing_channels': missing_ch
            })
            all_missing_channels.update(missing_ch)

    return {
        'patient_id': patient_id,
        'total_files': len(file_list),
        'valid_files': len(valid_files),
        'invalid_files': len(invalid_file_info),
        'valid_file_list': valid_files,
        'invalid_file_info': invalid_file_info,
        'all_missing_channels': sorted(list(all_missing_channels))
    }


def get_validation_summary(all_validation_results: List[Dict]) -> Dict:
    """Create a summary of validation results across all patients

    Args:
        all_validation_results: List of validation result dicts from validate_patient_files()

    Returns:
        Summary dictionary with aggregate statistics
    """
    total_files = sum(r['total_files'] for r in all_validation_results)
    total_valid = sum(r['valid_files'] for r in all_validation_results)
    total_invalid = sum(r['invalid_files'] for r in all_validation_results)

    # Collect all invalid files
    all_invalid_files = []
    for result in all_validation_results:
        for invalid_info in result['invalid_file_info']:
            all_invalid_files.append({
                'patient_id': result['patient_id'],
                'file': invalid_info['file'],
                'missing_channels': invalid_info['missing_channels']
            })

    # Count frequency of missing channels
    missing_channel_counts = {}
    for invalid_file in all_invalid_files:
        for channel in invalid_file['missing_channels']:
            missing_channel_counts[channel] = missing_channel_counts.get(channel, 0) + 1

    # Sort by frequency
    most_common_missing = sorted(
        missing_channel_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        'validation_enabled': not SKIP_CHANNEL_VALIDATION,
        'target_channels': TARGET_CHANNELS,
        'total_files_checked': total_files,
        'files_with_valid_channels': total_valid,
        'files_with_invalid_channels': total_invalid,
        'invalid_file_details': all_invalid_files,
        'missing_channel_frequency': dict(most_common_missing),
        'patients_processed': len(all_validation_results)
    }
