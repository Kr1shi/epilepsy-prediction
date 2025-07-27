import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy import signal

os.makedirs('visualize', exist_ok=True)

# Read EDF file
f = pyedflib.EdfReader('physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf')

n_channels = f.signals_in_file
signal_labels = f.getSignalLabels()
sigbufs = np.zeros((n_channels, f.getNSamples()[0]))

for i in range(n_channels):
    sigbufs[i, :] = f.readSignal(i)
    sampling_rate = f.getSampleFrequency(0)
f.close()

# Bandpass filter (0.5-50 Hz typical for epilepsy)
low_freq, high_freq = 0.5, 50
nyquist = sampling_rate / 2
low = low_freq / nyquist
high = high_freq / nyquist
b, a = signal.butter(4, [low, high], btype='band')
filtered_data = signal.filtfilt(b, a, sigbufs, axis=1)
# Notch filter (remove 50/60 Hz power line noise)
f0 = 60  # 50 Hz for Europe, 60 Hz for US
Q = 30
b_notch, a_notch = signal.iirnotch(f0, Q, sampling_rate)
filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data, axis=1)

# Normalize the filtered data (z-score normalization per channel)
normalized_data = np.zeros_like(filtered_data)
for ch in range(n_channels):
    normalized_data[ch, :] = (filtered_data[ch, :] - filtered_data[ch, :].mean()) / filtered_data[ch, :].std()

# Calculate duration and create time axis
duration = sigbufs.shape[1] / sampling_rate
time = np.linspace(0, duration, sigbufs.shape[1])

# Show first 10 seconds for clarity
end_sample = int(10 * sampling_rate)
time_short = time[:end_sample]

# Pick a few channels to compare
channels_to_plot = [0, 1, 2]

fig, axes = plt.subplots(len(channels_to_plot), 3, figsize=(20, 8))

for i, ch in enumerate(channels_to_plot):
    # Original data
    axes[i, 0].plot(time_short, sigbufs[ch, :end_sample])
    axes[i, 0].set_title(f'Original - {signal_labels[ch]}')
    axes[i, 0].set_ylabel('Amplitude (μV)')
    axes[i, 0].grid(True, alpha=0.3)
    
    # Bandpass filtered data
    axes[i, 1].plot(time_short, filtered_data[ch, :end_sample])
    axes[i, 1].set_title(f'Bandpass Filtered - {signal_labels[ch]}')
    axes[i, 1].set_ylabel('Amplitude (μV)')
    axes[i, 1].grid(True, alpha=0.3)
    
    # Normalized data
    axes[i, 2].plot(time_short, normalized_data[ch, :end_sample])
    axes[i, 2].set_title(f'Normalized - {signal_labels[ch]}')
    axes[i, 2].set_ylabel('Z-score')
    axes[i, 2].grid(True, alpha=0.3)

# Add x-labels to bottom plots
for col in range(3):
    axes[-1, col].set_xlabel('Time (seconds)')

plt.tight_layout()
plt.savefig('visualize/preprocessing_stages.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Original data range: {sigbufs.min():.2f} to {sigbufs.max():.2f}")
print(f"Filtered data range: {filtered_data.min():.2f} to {filtered_data.max():.2f}")
print(f"Normalized data range: {normalized_data.min():.2f} to {normalized_data.max():.2f}")
print("Preprocessing comparison saved to 'visualize/preprocessing_stages.png'")
