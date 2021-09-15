from scipy import signal
import numpy as np
import math
from matplotlib import pyplot as plt

T = np.linspace(-5, 5, 500)

sin_wave = np.sin(2*math.pi * T)
noise = 0.5*np.random.normal(0, 0.3, 500)

fc = 5  # Cut-off frequency
fs = 50  # Sampling Frequency

# Normalize the frequency
w = fc / (fs / 2) # Normalized Frequency w

# Apply the filter

b, a = signal.butter(5, w, 'low', analog=False)

print(sin_wave)
noisy_sin_wave = sin_wave + noise


output = signal.filtfilt(b, a, noisy_sin_wave)


plt.rcParamsDefault['lines.marker'] = '^'
plt.rcParamsDefault['lines.markersize'] = 10

plt.plot(T, noisy_sin_wave, label='noisy')
plt.plot(T, sin_wave, label='original')
plt.plot(T, output, label='filtered')

plt.legend()
plt.show()
