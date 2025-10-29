import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- Import functions from the pyceps library ---
try:
    from pyceps.pyceps.core import rceps, cepsf0, cepsenv
except ImportError:
    print("ERROR: Could not import pyceps library.")
    print("Please ensure it is installed correctly.")
    exit()

# -----------------------------------------------------------------
# TODO: CONFIGURE THESE PARAMETERS
# -----------------------------------------------------------------

# 1. Point this to one of your vowel recordings
AUDIO_FILE = 'C:/Users/ashut/OneDrive/Desktop/speech_processing_assn_1/recordings/hindi_vowel_aa.wav'  # <--- CHANGE THIS

# 2. Analysis parameters
SR = 44100              # Sampling rate
FRAME_LEN = 1024        # Frame length in samples
HOP_LEN = FRAME_LEN // 4  # Hop length (256 samples)
# --- Set to 6 frames to match your reference code ---
N_FRAMES_TO_ANALYZE = 6 

# 3. Start frame
START_FRAME_IDX = 129

# 4. Cepstral liftering cutoff
LIFTER_N = 40

# 5. Pitch (F0) search range in Hz
PITCH_FMIN = 70
PITCH_FMAX = 500

# -----------------------------------------------------------------
# SCRIPT LOGIC (Data Calculation - Same as before)
# -----------------------------------------------------------------

print(f"Loading audio file: {AUDIO_FILE}...")
try:
    y, sr = librosa.load(AUDIO_FILE, sr=SR)
except FileNotFoundError:
    print(f"ERROR: Audio file not found at '{AUDIO_FILE}'")
    print("Please update the AUDIO_FILE variable in the script.")
    exit()

# 1. STFT
S_complex = librosa.stft(y,
                         n_fft=FRAME_LEN,
                         hop_length=HOP_LEN,
                         win_length=FRAME_LEN,
                         window='hamming',
                         center=False)

# 2. Log-magnitude spectrogram
S_mag = np.abs(S_complex)
D_all_frames = np.log(S_mag + 1e-9)

# 3. Select frames
n_total_frames = D_all_frames.shape[1]
end_frame_idx = START_FRAME_IDX + N_FRAMES_TO_ANALYZE

if end_frame_idx > n_total_frames:
    print(f"Warning: Not enough frames. Adjusting N_FRAMES_TO_ANALYZE.")
    N_FRAMES_TO_ANALYZE = n_total_frames - START_FRAME_IDX
    end_frame_idx = n_total_frames

D_slice = D_all_frames[:, START_FRAME_IDX:end_frame_idx]
print(f"Analyzing {D_slice.shape[1]} frames (from index {START_FRAME_IDX} to {end_frame_idx-1})...")

# --- Use pyceps functions ---
f0_values, _ = cepsf0(D_slice, sr=SR, fmax=PITCH_FMAX, fmin=PITCH_FMIN, verbose=False)
quef, C_slice = rceps(D_slice, sr=SR) # quef is in seconds, C_slice is (FRAME_LEN, N_FRAMES)
envelope_log_slice = cepsenv(C_slice, lift_th=LIFTER_N)

# --- Find Formants (Same as before) ---
all_formants = []
freq_axis = librosa.fft_frequencies(n_fft=FRAME_LEN, sr=SR)

for i in range(N_FRAMES_TO_ANALYZE):
    env_log_frame = envelope_log_slice[:, i]
    peaks_indices, _ = find_peaks(env_log_frame, height=None, distance=5)
    formant_freqs = sorted(freq_axis[peaks_indices])
    
    if len(formant_freqs) < 3:
        formant_freqs.extend([np.nan] * (3 - len(formant_freqs)))
    
    all_formants.append(formant_freqs[:3])

# -----------------------------------------------------------------
# Generate Plots (NEW - Based on your reference code)
# -----------------------------------------------------------------
print("Generating stacked plots (user reference style)...")

# --- 1. Setup Plot Window and Offsets ---
fig, axs = plt.subplots(1, 2, figsize=(18, 9)) 
fig.suptitle(f'Cepstral Analysis: {N_FRAMES_TO_ANALYZE} Consecutive Frames for "{AUDIO_FILE}"', fontsize=16)

# Spectral offset (in dB) from your reference
SPECTRA_OFFSET = 70

# --- Dynamically calculate Cepstra Offset ---
# This finds the max peak in the valid pitch range to set a reasonable
# vertical offset, since our C_slice (real cepstrum) has a different
# scale than your reference code's "dB cepstrum".
PITCH_MIN_BIN = int(SR / PITCH_FMAX)
PITCH_MAX_BIN = int(SR / PITCH_FMIN)
PITCH_MIN_BIN = max(10, PITCH_MIN_BIN) # Avoid DC
PITCH_MAX_BIN = min(C_slice.shape[0], PITCH_MAX_BIN)

try:
    max_cep_peak = np.max(np.abs(C_slice[PITCH_MIN_BIN:PITCH_MAX_BIN, :]))
    # Stack plots by 2x the max pitch peak for clear separation
    CEPSTRA_OFFSET = max_cep_peak * 2.0 
    if CEPSTRA_OFFSET < 0.01: CEPSTRA_OFFSET = 0.1 # Handle silence
except ValueError: # Handle empty slice
    CEPSTRA_OFFSET = 0.1
print(f"Using dynamic cepstra offset: {CEPSTRA_OFFSET:.2f}")


# Lists for formant contours (from your reference)
f1_contour_points = []
f2_contour_points = []
f3_contour_points = []

# Get data axes
# Use bins (samples) for x-axis per your reference code
quef_bins = np.arange(C_slice.shape[0]) 
# Get pitch lags in bins (samples)
pitch_lags_samples = np.round(SR / f0_values).astype(int)


# --- 2. Loop Through Frames and Plot ---
for i in range(N_FRAMES_TO_ANALYZE):
    # --- Get data for frame 'i' ---
    spec_log = D_slice[:, i]
    spec_db = librosa.amplitude_to_db(np.exp(spec_log)) # Convert to dB for plotting
    
    env_log = envelope_log_slice[:, i]
    env_db = librosa.amplitude_to_db(np.exp(env_log)) # Convert to dB for plotting
    
    cepstrum_frame = C_slice[:, i]
    formants_frame = all_formants[i] # [f1, f2, f3] in Hz
    pitch_lag_frame = pitch_lags_samples[i]

    # --- Plot 1: Spectra (Left) ---
    y_offset_spec = i * SPECTRA_OFFSET
    axs[0].plot(freq_axis, spec_db + y_offset_spec, 'k-', linewidth=0.7, alpha=0.8, label=None)
    axs[0].plot(freq_axis, env_db + y_offset_spec, 'r-', linewidth=1.5, label='Envelope' if i == 0 else None)
    # Adjust x-position of text to be visible
    axs[0].text(-200, y_offset_spec, f'{i+1}', horizontalalignment='center', verticalalignment='center')

    # Store points for contour plot
    # Store points for contour plot (Corrected)
    # We find the y-value of the peak on the envelope curve
    if len(formants_frame) > 0 and not np.isnan(formants_frame[0]):
        f1_hz = formants_frame[0]
        # Find the index of the envelope closest to this frequency
        f1_index = np.argmin(np.abs(freq_axis - f1_hz))
        # Get the envelope's dB value at that index and add the offset
        y_peak = env_db[f1_index] + y_offset_spec
        f1_contour_points.append((f1_hz, y_peak))

    if len(formants_frame) > 1 and not np.isnan(formants_frame[1]):
        f2_hz = formants_frame[1]
        f2_index = np.argmin(np.abs(freq_axis - f2_hz))
        y_peak = env_db[f2_index] + y_offset_spec
        f2_contour_points.append((f2_hz, y_peak))

    if len(formants_frame) > 2 and not np.isnan(formants_frame[2]):
        f3_hz = formants_frame[2]
        f3_index = np.argmin(np.abs(freq_axis - f3_hz))
        y_peak = env_db[f3_index] + y_offset_spec
        f3_contour_points.append((f3_hz, y_peak))

    # --- Plot 2: Cepstra (Right) ---
    y_offset_cepstra = i * CEPSTRA_OFFSET
    axs[1].plot(quef_bins, cepstrum_frame + y_offset_cepstra, 'k-', linewidth=0.7, label=None)
    
    # Plot pitch peak (red dot)
    # Check if lag is in the valid range
    if pitch_lag_frame > PITCH_MIN_BIN and pitch_lag_frame < PITCH_MAX_BIN:
         # Check if it's actually a positive peak
        if cepstrum_frame[pitch_lag_frame] > 0:
             axs[1].plot(pitch_lag_frame, cepstrum_frame[pitch_lag_frame] + y_offset_cepstra, 
                         'ro', markersize=4, alpha=0.7, label='Pitch Peak' if i == 0 else None)
    
    axs[1].text(-30, y_offset_cepstra, f'{i+1}', horizontalalignment='center', verticalalignment='center') # Per your code


# --- 3. Finalize Plots (from your reference) ---
if f1_contour_points:
    axs[0].plot([p[0] for p in f1_contour_points], [p[1] for p in f1_contour_points], 'b-o', linewidth=2.5, markersize=3, alpha=0.7, label='F1')
if f2_contour_points:
    axs[0].plot([p[0] for p in f2_contour_points], [p[1] for p in f2_contour_points], 'b-o', linewidth=2.5, markersize=3, alpha=0.7, label='F2')
if f3_contour_points:
    axs[0].plot([p[0] for p in f3_contour_points], [p[1] for p in f3_contour_points], 'b-o', linewidth=2.5, markersize=3, alpha=0.7, label='F3')

axs[0].set_title("Short-Time Log Spectra")
axs[0].set_xlabel("Frequency (Hz)")
axs[0].set_ylabel("Frame Number / Amplitude (dB)")
axs[0].set_xlim(0, 5500) # From your script
axs[0].set_yticks([]) # From your script
axs[0].legend()
axs[0].grid(True)

axs[1].set_title("Short-Time Cepstra")
axs[1].set_xlabel("Quefrency (samples/bins)") # Clarified label
axs[1].set_ylabel("Frame Number / Amplitude")
axs[1].set_xlim(0, PITCH_MAX_BIN + 100) # Use our calculated MAX_LAG
axs[1].set_yticks([]) # From your script
axs[1].legend()
axs[1].grid(True)


# --- 4. Show and Save Plot ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
plt.savefig('stacked_cepstrum_analysis_vowel.png', bbox_inches='tight')
print("Saved 'stacked_cepstrum_analysis_vowel.png'")
plt.show()


# -----------------------------------------------------------------
# Report Averages (Same as before)
# -----------------------------------------------------------------
avg_pitch = np.nanmean(f0_values)
avg_formants = np.nanmean(all_formants, axis=0)

print("\n--- Average Values (Objective 2) ---")
print(f"Analyzed {N_FRAMES_TO_ANALYZE} frames from {AUDIO_FILE}")
print(f"Average Pitch (F0): {avg_pitch:.2f} Hz")
if len(avg_formants) >= 3:
    print(f"Average Formant 1 (F1): {avg_formants[0]:.2f} Hz")
    print(f"Average Formant 2 (F2): {avg_formants[1]:.2f} Hz")
    print(f"Average Formant 3 (F3): {avg_formants[2]:.2f} Hz")
print("--------------------------------------")