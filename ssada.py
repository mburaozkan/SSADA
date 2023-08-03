from scipy.signal import butter, sosfreqz, sosfilt, hilbert
import numpy as np
from oct_converter.readers import FDA
import matplotlib.pyplot as plt

def get_shape(oct_volume):
    num_b_scans = len(oct_volume.volume)
    num_a_scans = len(oct_volume.volume[0]) if num_b_scans > 0 else 0
    return num_b_scans, num_a_scans

def create_spectral_bands(oct_volume):
    low_cutoff = 5.0  # Lower cutoff frequency (Hz)
    high_cutoff = 200.0  # Upper cutoff frequency (Hz)
    order = 4  # Filter order

    spectral_bands = []
    num_b_scans, num_a_scans = get_shape(oct_volume)

    # Convert A-scan spacing to seconds
    a_scan_spacing_mm = 320 / num_a_scans  # 320 pixels for a scans
    a_scan_spacing_seconds = a_scan_spacing_mm / 1000.0

    # Calculate the sampling frequency (fs) based on A-scan spacing
    fs = 1.0 / a_scan_spacing_seconds

    # Normalize cutoff frequencies to the range 0 to 1
    low_cutoff_norm = low_cutoff / (fs / 2)
    high_cutoff_norm = high_cutoff / (fs / 2)

    print(f"Sampling Frequency (fs): {fs}")
    print(f"Normalized Low Cutoff Frequency (Wn_low): {low_cutoff_norm}")
    print(f"Normalized High Cutoff Frequency (Wn_high): {high_cutoff_norm}")

    # Design bandpass filter
    sos = butter(order, [low_cutoff_norm, high_cutoff_norm], btype='band', output='sos')

    # Apply bandpass filter to each OCT volume in the list
    spectral_band = [sosfilt(sos, volume.T).T for volume in oct_volume.volume]
    spectral_bands.append(spectral_band)

    return spectral_bands

def calculate_phase_variance(spectral_bands):
    phase_variances = []

    for spectral_band in spectral_bands:
        # Apply Hilbert transform to get complex-valued signal
        complex_signal = hilbert(spectral_band, axis=0)

        # Extract phase information
        phase = np.angle(complex_signal)

        # Calculate phase variance between consecutive B-scans
        phase_diff = np.diff(phase, axis=0)
        phase_variance = np.var(phase_diff, axis=0)

        phase_variances.append(phase_variance)

    return phase_variances


def generate_oct_angiogram(phase_variances):
    angiogram = np.sum(phase_variances, axis=0)
    return angiogram

# Read the OCT data
# filepath = "./Sample OCT/ACAR DILEK/acar, dilek _11018_Enhanced HD Line_OD_2021-07-07_09.09.34_1.OCT"
# poct = (filepath)
# oct_volumes = poct.read_oct_volume()


filepath = "./192784.fda"
fda = FDA(filepath)

oct_volumes = fda.read_oct_volume()

# Step 1: Split-Spectrum Processing
spectral_bands = create_spectral_bands(oct_volumes)

# Step 2: Phase-Variance Analysis
phase_variances = calculate_phase_variance(spectral_bands)

# Step 3: Combine Results to generate OCT angiogram
angiogram_result = generate_oct_angiogram(phase_variances)

# Display the OCT angiogram
plt.imshow(angiogram_result, cmap='gray')
plt.title('SSADA Angiogram')
plt.colorbar(label='Flow Intensity')
plt.show()
