"""
PHY 407: Computational Physics, Lab 05, Question 03
Author: John Wood
Date: October 15, 2022

This script find the Fourier transform for sea level pressure (SLP) data with the FFT algorithm (Numpy
implementation). Raw SLP data, and longitudinal wavenumbers 3 and 5 are plotted.

Outputs:
    - Lab04_Q03_a_wavenumber{i}.png for i in {3, 5}: The SLP, i-th longitudinal wavenumber.
    - Lab04_Q03_a_raw.png: Raw SLP data

"""
import numpy as np
import matplotlib.pyplot as plt

# ============ PSEUDOCODE ==============================================================================================
# Define figure size, dpi and font size for plots
# Define list of wavenumbers to analyze
#
# Load in longitude, time, and SLP data as Numpy arrays
# Compute FFT of SLP data
#
# Define function to plot SLP data
#     Set SLP data, title suffix and filename suffix
#     Create new figure
#     Plot SLP data vs (time and longitude) with filled contours
#     Set x and y labels for plot
#     Set plot title and append title suffix
#     Save plot, appending filename suffix to the filename
#
# For each wavenumber in previously defined list:
#     Create empy arrays of zeros the same shape as the FFT array
#     Set the (wavenumber - 1)-th column to that column from the FFT array (for each time row, take only certain
#       column wavenumber)
#     Compute and plot reverse FFT
#
# Plot raw SLP data
#
# ======================================================================================================================

# ============ CONSTANTS ===============================================================================================
# Plotting parameters
fig_size = (12, 8)
dpi = 300
plt.rcParams.update({'font.size': 12})

# Extract data
longitude = np.loadtxt('lon.txt') # degrees
time = np.loadtxt('times.txt') # days since January 1, 2015
slp = np.loadtxt('SLP.txt') # axes are (time, longitude)

analyzed_wavenumbers = (3, 5)

# ============ QUESTION 2A =============================================================================================
print('============ QUESTION 2A ======================================================================================')


def plot_slp_data(slp_data, title_suffix='', filename_suffix=''):
    # Plot data
    plt.figure(figsize=fig_size, dpi=dpi)
    plt.contourf(longitude, time, slp_data)

    # Finalize and save plot
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Days since Jan. 1, 2015')
    plt.title('SLP anomaly (hPa)' + title_suffix)
    plt.colorbar()
    plt.savefig(f'Lab04_Q03_a_{filename_suffix}.png')


transform = np.fft.rfft(slp)

for wavenumber in analyzed_wavenumbers:
    print(f'Finding and plotting wavenumber {wavenumber}...')

    # Set required coefficients to zero
    wavenumber_transform = np.zeros(np.shape(transform), complex)
    wavenumber_transform[:, wavenumber - 1] = transform[:, wavenumber - 1]

    # Recover wave
    wavenumber_slp = np.fft.irfft(wavenumber_transform)

    # Plot and save wave
    plot_slp_data(wavenumber_slp, f' (Wavenumber {wavenumber})', f'wavenumber{wavenumber}')

    print('Done')

# Plot raw data
print('Plotting raw data...')
plot_slp_data(slp, '', 'raw')
print('Done')
