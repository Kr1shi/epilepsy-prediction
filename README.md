# Epilepsy Prediction: Data Preparation

This document outlines the steps to process raw EEG data from the CHB-MIT Scalp EEG Database into a format suitable for training a machine learning model for epilepsy prediction.

The data preparation is a two-step process:
1.  **Data Segmentation**: Identifying and labeling segments of EEG data as "preictal" (before a seizure) or "interictal" (between seizures).
2.  **Data Preprocessing**: Cleaning, transforming, and saving the segmented data as spectrograms in HDF5 files.

## Prerequisites

*   Python 3.x
*   Required Python packages (e.g., `mne`, `numpy`, `h5py`, `tqdm`, `scikit-learn`). You can install them using pip:
    ```bash
    pip install mne numpy h5py tqdm scikit-learn
    ```
*   The CHB-MIT dataset downloaded and extracted into the `physionet.org/files/chbmit/1.0.0/` directory.

## Step 1: Data Segmentation

The `data_segmentation.py` script reads the summary files for each patient, identifies seizure events, and creates labeled segments of data.

### How to Run

To run the data segmentation process, execute the following command in your terminal:

```bash
python data_segmentation.py
```

### What it Does

*   Iterates through each patient's summary file (`-summary.txt`).
*   Parses the seizure times and file information.
*   Creates "preictal" segments in the time window leading up to each seizure.
*   Creates "interictal" segments from periods of normal brain activity, ensuring they do not overlap with preictal periods.
*   Saves all segment information into a single JSON file: `all_patients_segments.json`.

### Configuration

The behavior of the segmentation can be modified by editing the parameters in `data_segmentation_helpers/config.py`:

*   `PREICTAL_WINDOW`: The duration (in seconds) before a seizure to label as preictal.
*   `SEGMENT_DURATION`: The length (in seconds) of each individual data segment.
*   `INTERICTAL_POST_SEIZURE_GAP`: A buffer time (in seconds) after a seizure to avoid including post-seizure effects in interictal data.

## Step 2 (Optional): Data Visualization

Before committing to the full preprocessing pipeline, you can visualize the output of a few sample segments to ensure the signal processing is working as expected. The `data_visualization.py` script is designed for this purpose.

### How to Run

First, ensure you have `matplotlib` installed:

```bash
pip install matplotlib
```

Then, run the visualization script:

```bash
python data_visualization.py
```

### What it Does

*   Randomly selects a few preictal and interictal segments from `all_patients_segments.json`.
*   Applies the same preprocessing steps as the main pipeline (filtering, normalization, STFT, etc.) to these segments.
*   Generates and saves PNG images of the resulting spectrograms in the `preprocessing/visualizations/` directory.

This allows you to quickly inspect the quality of the processed data and verify that your configuration is correct before running the full, time-consuming preprocessing step.

## Step 3: Data Preprocessing

The `data_preprocessing.py` script takes the `all_patients_segments.json` file generated in the previous step and processes the raw EEG data for each segment.

### How to Run

To run the data preprocessing pipeline, execute the following command:

```bash
python data_preprocessing.py
```

### What it Does

*   **Loads Segments**: Reads the segment information from `all_patients_segments.json`.
*   **Validates Channels**: Ensures that all required EEG channels are present for each segment.
*   **Balances and Splits Data**: Balances the number of preictal and interictal segments and splits them into training, validation, and test sets.
*   **Signal Processing**: For each segment, it performs the following steps:
    1.  Selects target EEG channels.
    2.  Detects and interpolates bad channels.
    3.  Removes signal artifacts.
    4.  Applies band-pass and notch filters.
    5.  Normalizes the signal.
    6.  Computes a spectrogram using a Short-Time Fourier Transform (STFT).
*   **Saves Data**: Saves the processed spectrograms and their corresponding labels and metadata into HDF5 files (`train_dataset.h5`, `val_dataset.h5`, `test_dataset.h5`) in the `preprocessing/data/` directory.

### Features

*   **Checkpointing**: The script saves its progress in `preprocessing/checkpoints/progress.json`. If the script is interrupted, it can be resumed from where it left off.
*   **Incremental Saving**: Data is saved to the HDF5 files in batches, making the process robust to crashes.
*   **Logging**: Detailed logs of the preprocessing pipeline are saved in `preprocessing/logs/preprocessing.log`.

### Configuration

The preprocessing pipeline can be configured by editing the parameters in `data_segmentation_helpers/config.py`:

*   `TARGET_CHANNELS`: The list of EEG channels to use.
*   `LOW_FREQ_HZ`, `HIGH_FREQ_HZ`, `NOTCH_FREQ_HZ`: Parameters for the frequency filters.
*   `STFT_NPERSEG`, `STFT_NOVERLAP`: Settings for the STFT spectrogram generation.
*   `SKIP_CHANNEL_VALIDATION`: Set to `True` to skip the initial channel validation step for faster startup on subsequent runs.
