import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pyceps.pyceps.core import rceps, cepsf0, cepsenv

# Config
AUDIO_FILE = "recordings/hindi_vowel_aa.wav"

SR = 44100                # Sampling rate
FRAME_LEN = 1024          # Frame length (samples)
HOP_LEN = FRAME_LEN // 4  # Hop length (256 samples)
START_FRAME_IDX = 129     # Start index for 6-frame analysis
N_FRAMES_TO_ANALYZE = 6

LIFTER_N = 40             # Cepstral liftering cutoff
PITCH_FMIN, PITCH_FMAX = 70, 500  # F0 search range (Hz)

# Load and pre-process audio
print(f"\nLoading audio file: {AUDIO_FILE}")
try:
    y, sr = librosa.load(AUDIO_FILE, sr=SR)
except FileNotFoundError:
    raise FileNotFoundError(f"Audio file not found at {AUDIO_FILE}")

# Compute STFT (Hamming window)
S_complex = librosa.stft(
    y, n_fft=FRAME_LEN, hop_length=HOP_LEN,
    win_length=FRAME_LEN, window="hamming", center=False
)

# Log-magnitude spectrogram
S_mag = np.abs(S_complex)
D_all_frames = np.log(S_mag + 1e-9)

# Frame selection
end_frame_idx = START_FRAME_IDX + N_FRAMES_TO_ANALYZE
n_total_frames = D_all_frames.shape[1]

if end_frame_idx > n_total_frames:
    end_frame_idx = n_total_frames
    N_FRAMES_TO_ANALYZE = n_total_frames - START_FRAME_IDX
    print("Warning: Adjusted number of frames due to end-of-signal limit.")

D_slice = D_all_frames[:, START_FRAME_IDX:end_frame_idx]
print(f"Analyzing {D_slice.shape[1]} frames (indices {START_FRAME_IDX}–{end_frame_idx - 1})")


# Cepstral analysis
f0_values, _ = cepsf0(D_slice, sr=SR, fmin=PITCH_FMIN, fmax=PITCH_FMAX, verbose=False)
quef, C_slice = rceps(D_slice, sr=SR)
envelope_log_slice = cepsenv(C_slice, lift_th=LIFTER_N)


# Formant estimation
freq_axis = librosa.fft_frequencies(n_fft=FRAME_LEN, sr=SR)
all_formants = []

for i in range(N_FRAMES_TO_ANALYZE):
    env_log = envelope_log_slice[:, i]
    peaks, _ = find_peaks(env_log, distance=5)
    formant_freqs = sorted(freq_axis[peaks])[:3]
    while len(formant_freqs) < 3:
        formant_freqs.append(np.nan)
    all_formants.append(formant_freqs)

# Visualization
print("Generating framewise stacked plots...")

fig, axs = plt.subplots(1, 2, figsize=(18, 9))
fig.suptitle(f"Cepstral Analysis ({N_FRAMES_TO_ANALYZE} frames): {AUDIO_FILE}", fontsize=15)

SPECTRA_OFFSET = 70  # for stacked spectra display

# Compute dynamic cepstra offset
PITCH_MIN_BIN = int(SR / PITCH_FMAX)
PITCH_MAX_BIN = int(SR / PITCH_FMIN)
PITCH_MIN_BIN = max(10, PITCH_MIN_BIN)
PITCH_MAX_BIN = min(C_slice.shape[0], PITCH_MAX_BIN)

try:
    max_cep_peak = np.max(np.abs(C_slice[PITCH_MIN_BIN:PITCH_MAX_BIN]))
    CEPSTRA_OFFSET = max(0.1, 2.0 * max_cep_peak)
except ValueError:
    CEPSTRA_OFFSET = 0.1

print(f"Using cepstra offset: {CEPSTRA_OFFSET:.2f}")

f1_points, f2_points, f3_points = [], [], []
quef_bins = np.arange(C_slice.shape[0])
pitch_lags = np.round(SR / f0_values).astype(int)

# Framewise plots
for i in range(N_FRAMES_TO_ANALYZE):
    spec_log = D_slice[:, i]
    env_log = envelope_log_slice[:, i]
    cepstrum = C_slice[:, i]
    formants = all_formants[i]
    pitch_lag = pitch_lags[i]

    spec_db = librosa.amplitude_to_db(np.exp(spec_log))
    env_db = librosa.amplitude_to_db(np.exp(env_log))

    # --- Plot 1: Spectral envelopes ---
    y_off = i * SPECTRA_OFFSET
    axs[0].plot(freq_axis, spec_db + y_off, "k-", lw=0.7, alpha=0.8)
    axs[0].plot(freq_axis, env_db + y_off, "r-", lw=1.4)
    axs[0].text(-200, y_off, f"{i + 1}", ha="center", va="center")

    # Mark formants
    for j, (fset, contour) in enumerate(zip(formants, [f1_points, f2_points, f3_points])):
        if not np.isnan(fset):
            idx = np.argmin(np.abs(freq_axis - fset))
            contour.append((fset, env_db[idx] + y_off))

    # --- Plot 2: Cepstra ---
    y_off_c = i * CEPSTRA_OFFSET
    axs[1].plot(quef_bins, cepstrum + y_off_c, "k-", lw=0.7)
    if PITCH_MIN_BIN < pitch_lag < PITCH_MAX_BIN and cepstrum[pitch_lag] > 0:
        axs[1].plot(pitch_lag, cepstrum[pitch_lag] + y_off_c, "ro", ms=4, alpha=0.7)
    axs[1].text(-30, y_off_c, f"{i + 1}", ha="center", va="center")

# Plot formant contours
for points, label in zip([f1_points, f2_points, f3_points], ["F1", "F2", "F3"]):
    if points:
        axs[0].plot(*zip(*points), "b-o", lw=2.0, ms=3, alpha=0.7, label=label)

# Finalize figure
axs[0].set(
    title="Short-Time Log Spectra",
    xlabel="Frequency (Hz)",
    ylabel="Frame Number / Amplitude (dB)",
    xlim=(0, 5500),
)
axs[0].set_yticks([])
axs[0].legend()
axs[0].grid(True)

axs[1].set(
    title="Short-Time Cepstra",
    xlabel="Quefrency (samples/bins)",
    ylabel="Frame Number / Amplitude",
    xlim=(0, PITCH_MAX_BIN + 100),
)
axs[1].set_yticks([])
axs[1].legend()
axs[1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("stacked_cepstrum_analysis_vowel.png", bbox_inches="tight")
print("Saved plot as 'stacked_cepstrum_analysis_vowel.png'")
plt.show()


# Summary statistics

avg_pitch = np.nanmean(f0_values)
avg_formants = np.nanmean(all_formants, axis=0)

print("\n--- Average Acoustic Parameters ---")
print(f"Frames analyzed: {N_FRAMES_TO_ANALYZE}")
print(f"Average Pitch (F0): {avg_pitch:.2f} Hz")
print(f"Formant 1 (F1): {avg_formants[0]:.2f} Hz")
print(f"Formant 2 (F2): {avg_formants[1]:.2f} Hz")
print(f"Formant 3 (F3): {avg_formants[2]:.2f} Hz")
print("----------------------------------\n")
