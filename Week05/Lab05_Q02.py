"""
PHY 407: Computational Physics, Lab 05, Question 02
Author: John Wood
Date: October 16, 2022

Loads the GraviteaTime.wav file, visualizes its channels over time and over a short interval of around 35 ms. Using the
FFT algorithm (Numpy implementation), a filter sets all frequencies above 880 Hz to 0. The unfiltered and filtered
Fourier coefficients and audio signals are plotted for each channel. The resulting audio data is outputted as a .wav
file.

Outputs:
    - Lab05_Q02_b.png: Audio signal for both channels over time.
    - Lab05_Q02_c.png: Audio signal for both channels over short time interval.
    - Lab05_Q02_d_channel0.png: Unfiltered and filtered Fourier coefficients and signal for channel 0
    - Lab05_Q02_d_channel1.png: Same as above, but for channel 1
    - GraviteaTime_lpf.wav: The filtered audio

"""

# ============ PSEUDOCODE ==============================================================================================
# Set plotting settings (figure size, resolution, and font size)
# Read and store audio data and sample rate from given .wav file
# Separate audio data into channels
# Calculate sample time, dt, as the inverse of sample frequency
#
# Set the time axis from 0 to time of .wav, with same number of elements as in either channel
# Define duration of short time interval
# Compute index in time axis corresponding to the end of that interval
#
# ------------ Question 2B ---------------------------------------------------------------------------------------------
# Initialize figure with two subplots
# For each channel:
#     Plot the channel over time
# Add titles and labels to plot
# Save plot
#
# ------------ Question 2C ---------------------------------------------------------------------------------------------
# Initialize figure with two subplots
# For each channel:
#     Plot the channel over time, cut to index previously found for the end of the interval
# Add titles and labels to plot
# Save plot
#
# ------------ Question 2D ---------------------------------------------------------------------------------------------
# Define maximum allowed frequency
# Using number of audio samples and dt, find Fourier frequencies
# Compute index in frequencies corresponding to the first frequency over the maximum
# Create empty array for filtered audio data. Should have same dimensions and type as original audio data
#
# For each channel:
#     Compute the Fourier transform of the channel
#     Define filtered coefficients as above, but with zeros at and past the maximum frequency index found above
#     Compute filtered channel data with inverse Fourier transform of above coeffecients
#     Add filtered channel data to array defined previously
#
#     Plot Fourier coefficients vs. frequency
#     Plot filtered Fourier coefficients vs. frequency

#     Plot channel vs. time (limited to short time interval)
#     Plot filtered channel vs. time (limited to short time interval)
#
# ------------ Question 2E ---------------------------------------------------------------------------------------------
# Output filtered audio data as .wav file
# ======================================================================================================================

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# ============ CONSTANTS ===============================================================================================
# Plotting parameters
fig_size = (12, 10)
dpi = 300
plt.rcParams.update({'font.size': 15})

# Audio
sample_rate, audio_data = wav.read('GraviteaTime.wav')
samples = np.shape(audio_data)[0]
channels = (audio_data[:, 0], audio_data[:, 1])

dt = 1 / float(sample_rate)

time_axis = np.linspace(0, samples / sample_rate, samples) # seconds
short_duration = 35e-3 # seconds
short_time_idx = int(short_duration * sample_rate)

# ============ QUESTION 2B =============================================================================================
print('============ QUESTION 2B ======================================================================================')
# Plot channels vs. time over all time
fig, ax = plt.subplots(2, 1, figsize=fig_size, dpi=dpi, sharey=True, sharex=True)

for i, channel in enumerate(channels):
    print(f'Plotting channel {i}...')
    ax[i].plot(time_axis, channel, c=f'C{i}')
    ax[i].set_title(f'Channel {i}')
    ax[i].set_ylabel('Amplitude')
    print('Done')

ax[1].set_xlabel('Time (s)')

plt.savefig('Lab05_Q02_b.png')

# ============ QUESTION 2C =============================================================================================
print()
print('============ QUESTION 2C ======================================================================================')
# Plot channels vs. time over short period of time
fig, ax = plt.subplots(2, 1, figsize=fig_size, dpi=dpi, sharey=True, sharex=True)

for i, channel in enumerate(channels):
    print(f'Plotting channel {i}...')
    ax[i].plot(time_axis[:short_time_idx + 1] * 1e3, channel[:short_time_idx + 1], c=f'C{i}')
    ax[i].set_title(f'Channel {i}')
    ax[i].set_ylabel('Amplitude')
    print('Done')

ax[1].set_xlabel('Time (ms)')

plt.savefig('Lab05_Q02_c.png')

# ============ QUESTION 2D =============================================================================================
print()
print('============ QUESTION 2D ======================================================================================')
# Find maximum index of coefficients to accept
max_allowed_frequency = 880 # Hz
frequencies = np.fft.rfftfreq(samples, dt)
# Finds maximum index from frequencies where value is <= 880
max_allowed_index = np.asarray(frequencies <= max_allowed_frequency).nonzero()[0][-1]

filtered_audio_data = np.empty(np.shape(audio_data), dtype=audio_data.dtype)

for i, channel in enumerate(channels):
    print(f'Filtering channel {i}...')
    # Filter channel
    coefficients = np.fft.rfft(channel)

    filtered_coefficients = np.zeros(len(coefficients), complex)
    filtered_coefficients[:max_allowed_index + 1] = coefficients[:max_allowed_index + 1]

    filtered_channel = np.fft.irfft(filtered_coefficients)
    filtered_audio_data[:, i] = filtered_channel
    print('Done')

    # Plots
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=fig_size, dpi=dpi)

    # Plot Fourier coefficients
    ax0.plot(frequencies, abs(coefficients), label='Unfiltered')
    ax0.plot(frequencies, abs(filtered_coefficients), label='Filtered')

    ax0.set_title('Fourier coefficients')
    ax0.set_xlabel('Frequency (Hz)')
    ax0.set_ylabel('Amplitude')
    ax0.legend()

    print(f'Generating plots for channel {i}...')
    # Plot channel and filtered channel (only over short period of time)
    ax1.plot(time_axis[:short_time_idx + 1] * 1e3, channel[: short_time_idx + 1], label='Unfiltered', c='C2')
    ax1.plot(time_axis[:short_time_idx + 1] * 1e3, filtered_channel[: short_time_idx + 1], label='Filtered', c='C3')

    ax1.set_title(f'Channel {i}')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.legend()

    plt.tight_layout()
    plt.savefig(f'Lab05_Q02_d_channel{i}.png')
    print('Done')

# ============ QUESTION 2E =============================================================================================
print()
print('============ QUESTION 2E ======================================================================================')
# Output filtered signal as wav file
wav.write('GraviteaTime_lpf.wav', sample_rate, filtered_audio_data)
print('Outputted .wav file')
