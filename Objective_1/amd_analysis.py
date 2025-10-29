import numpy as np
import matplotlib.pyplot as plt
import librosa

file_path = "recordings/hindi_vowel_aa.wav"
y, sr = librosa.load(file_path, sr=44100)
# Find a good starting point (e.g., 15000 samples in)
start_sample = 33700 
frame_size = 1024

frame = y[start_sample : start_sample + frame_size]
min_lag = 110
max_lag = 551

amdf_values = []

for lag in range(min_lag, max_lag):
    # Slice the frame to get the original and shifted versions
    original_frame = frame[lag:]
    shifted_frame = frame[:-lag]

    # Calculate the sum of absolute differences
    diff = np.abs(original_frame - shifted_frame)
    amdf_values.append(np.mean(diff)) 
    # Using mean makes it the "Average" MDF

# Convert list to a NumPy array for easier processing
amdf_values = np.array(amdf_values)
# Find the index of the minimum AMDF value
dip_index = np.argmin(amdf_values)

# The actual lag is that index + our starting lag
pitch_period_samples = min_lag + dip_index
pitch_hz = sr / pitch_period_samples

print(f"Calculated Pitch Period: {pitch_period_samples} samples")
print(f"Calculated Pitch Frequency: {pitch_hz:.2f} Hz")
# Create an array of the lags you tested
lags = np.arange(min_lag, max_lag)

plt.figure(figsize=(10, 4))
plt.plot(lags, amdf_values)
plt.title('Average Magnitude Difference Function (AMDF)')
plt.xlabel('Lag (samples)')
plt.ylabel('Average Difference')

# Mark the dip you found
plt.axvline(x=pitch_period_samples, color='red', linestyle='--', 
            label=f'Pitch Period ({pitch_period_samples} samples)')
plt.legend()
plt.grid(True)

plt.show()
