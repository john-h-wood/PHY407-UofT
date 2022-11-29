"""
PHY 407: Computational Physics, Lab 09, Question 01
Author: John Wood
Date: November 28, 2022

In this script, we use the Crank-Nicholson algorithm to solve for the wavefunction of an electron in an infinite
square well over time. Initial conditions are set, the psi is found over a set number of time steps, each separated
by dt seconds. The real part of psi is plotted at some defined times. The expectation value of position is found and
plotted against time. The normalization of psi is found and plotted over time. Energy is found over time. Both of
these quantities should be conserved, and statistics to check this are printed.

Outputs:
    - Lab09_Q01_b_9.png: psi at 0*T where T is the final time
    - Lab09_Q01_b_0.25.png
    - Lab09_Q01_b_0.5.png
    - Lab09_Q01_b_0.75.png
    - Lab09_Q01_b_1.png
    - Lab09_Q01_b_trajectory.png: Expectation value of position over time

    - Lab09_Q01_c_norm.png: Normalization of psi over time
    - Lab09_Q01_c_energy.png: Energy of psi over time

    - Printed: progress updates
    - Printed: Greatest absolute error of normalization
    - Printed: Range, std, and mean of energy

"""

# ============ PSEUDOCODE ==============================================================================================
# import functions for integration, numpy, plotting, and linear algebra
# Define plotting preferences
# Define given problem constants (potential function, phi(t=0), L, m, space and time axes)
#
# Define array to store phi over time with dimensions (time steps, space steps)
# Set initial conditions in this array
# Set phi to be 0 at boundaries (since potential is infinite there)
#
# ------------ Question 1A ---------------------------------------------------------------------------------------------
# From problem constants, calculate the discretized Hamiltonian
# Use the Crank-Nicholson algorithm to step phi forward in time, saving results:
#   For each time step but the last:
#       Find psi at next time step (only non-boundary points) using matrix multiplication from lab handout and LU
#       decomposition
#
# ------------ Question 1B ---------------------------------------------------------------------------------------------
# For each required psi plotting time:
#   Plot the real part of psi at that time
#
# Define empty array for expectation value of position over time with length [time steps]
# For each time step: populate array using Equation 4 from lab handout. Take only real part of integral result
# Plot expectation value of position over time
#
# ------------ Question 1C ---------------------------------------------------------------------------------------------
# Create empty `normalization over time' array with length [time steps]
# For each time step: populate array using Equation 3 from lab handout (except the left hand side might not be 1).
#   Take only real part of integral result
# Print greatest difference of normalization from 1
# Plot normalization over time
#
# Create empty array for energy over time with length [time steps]
# For each time step:
#   Populate array using Equation 4 from lab handout and Hamiltonian from Question 1A
#   Note: H_D only works for non-boundary points. Operated psi's must be padded on both sides with a zero.
# Print energy range, STD and mean
# Plot energy over time
#
# ======================================================================================================================

import numpy as np
import scipy.constants as spc
import matplotlib.pyplot as plt
from math import sqrt, pi, floor
from scipy.integrate import simpson
from scipy.linalg import lu_factor, lu_solve

# ============ SETUP ===================================================================================================
print('============ SETUP ============================================================================================')
# Plotting preferences
fig_size = (6.4, 4.8)
dpi = 300
plt.rcParams.update({'font.size': 13})
plotting_fracs = (0, 0.25, 0.5, 0.75, 1) # fractions of final integration time

# Physical constants
m = 9.109e-31 # kg
L = 1e-8 # m

sigma = L / 25
kappa = 500 / L
x_0 = L / 5
phi_0 = sqrt(1 / (abs(sigma) * sqrt(2 * pi)))

# Integration constants
spatial_segments = 1024 # should be even and greater than four
time_steps = 3000

# Integration segments without boundary points
non_bound_segments = 1024 - 2

dx = L / spatial_segments # m
dt = 1e-18 # s

space_axis = np.linspace(-L / 2, L / 2, spatial_segments)
# The line below is perhaps inefficient (I could have used np.arange), but using a cumulative sum ensures the
# separation between time steps is (more or less) constant and there is exactly the desired quantity of time steps
time_axis = np.cumsum(np.ones(time_steps) * dt)

# More physical constants
phi_t_0 = phi_0 * np.exp(-((space_axis - x_0) ** 2 / (4 * sigma ** 2)) + (complex(0, 1) * kappa * space_axis))
potential = 0 # (since we are not concerned with the boundaries)

# Results array
phi = np.empty((time_steps, spatial_segments), complex)
# Set initial conditions
phi[0, :] = phi_t_0
# Set known conditions
phi[:, 0] = complex(0, 0)
phi[:, -1] = complex(0, 0)

print('Done setup of constants.')

# ============ QUESTION 1A =============================================================================================
print()
print('============ QUESTION 1A ======================================================================================')
print('Calculating H_D and doing LU decomposition...')
# Calculate the discretized Hamiltonian, L matrix and R matrix
A = - (spc.hbar ** 2) / (2 * m * (dx ** 2))
# In general, B is not constants, but it is here since potential is constant where we are integrating (inside the
# spatial boundary)
B = potential - (2 * A)

H_D = (np.eye(non_bound_segments) * B) + A * (np.eye(non_bound_segments, k=-1) + np.eye(non_bound_segments, k=1))
next_matrix = np.eye(non_bound_segments) + (complex(0, 1) * (dt / (2 * spc.hbar)) * H_D)
previous_matrix = np.eye(non_bound_segments) - (complex(0, 1) * (dt / (2 * spc.hbar)) * H_D)

# Perform lu decomposition of next_matrix (L matrix)
lu, piv = lu_factor(next_matrix)
print('Done.\n')

# For each time step but the last, find psi at the next time
for n in range(time_steps - 1):
    if (n + 1) % 600 == 0:
        print(f'Setting phi at time step {n + 1} of {time_steps}')
    v = np.matmul(previous_matrix, phi[n, 1:-1])
    phi[n + 1, 1:-1] = lu_solve((lu, piv), v)
print('Done numerical integration.')

# ============ QUESTION 1B =============================================================================================
print()
print('============ QUESTION 1B ======================================================================================')
# Plot particle paths
for frac in plotting_fracs:
    time_index = floor((time_steps - 1) * frac)
    time = time_axis[time_index] # s

    plt.figure(figsize=fig_size, dpi=dpi)

    plt.plot(space_axis, np.real(phi[time_index, :]), label=r'$\Re(\phi)$')
    plt.axvline(L / 2, c='black', ls='--', label=r'$L/2, -L/2$')
    plt.axvline(- L / 2, c='black', ls='--')

    plt.title(rf'$\phi$ at $t=${time} s')
    plt.xlabel(r'$x$ (m)')
    plt.ylabel(r'$\Re(\phi)$')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'Lab09_Q01_b_{frac}.png')
    plt.close()

    print(f'Plotted phi at t = {frac} * T')

# Calculate expectation value of position
expected_position = np.empty(time_steps, float)
for t in range(time_steps):
    phi_t = phi[t, :]
    integrands = np.conj(phi_t) * (space_axis * phi_t)
    expected_position[t] = np.real(simpson(integrands, space_axis))

print()
print('Calculated expectation value of position.')

# Plot expectation value of position over time
plt.figure(figsize=fig_size, dpi=dpi)

plt.plot(time_axis, expected_position)
plt.axhline(L / 2, c='black', ls='--', label=r'$L/2, -L/2$')
plt.axhline(- L / 2, c='black', ls='--')

plt.title('Expectation value of position over time')
plt.xlabel(r'$t$ (s)')
plt.ylabel(r'$\langle x\rangle$ (m)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig(f'Lab09_Q01_b_tajectory.png')
plt.close()
print('Expectation value of position plotted.')

# ============ QUESTION 2C =============================================================================================
print()
print('============ QUESTION 2C ======================================================================================')

# Check normalization over time
print('Normalization -------------------------------------------------------------------------------------------------')
normalization = np.empty(time_steps, float)
for t in range(time_steps):
    phi_t = phi[t, :]
    # Note, the integrands must be real. This is assured by using np.abs
    integrands = np.abs(phi_t) ** 2
    normalization[t] = simpson(integrands, space_axis)

plt.figure(figsize=fig_size, dpi=dpi)
plt.plot(time_axis, normalization, label='Probability', c='red')
plt.axhline(1, c='black', ls='--', label='Expected')

plt.title(r'$P_{-\infty<x<\infty}$ over time')
plt.xlabel('Time (s)')
plt.ylabel(r'$P_{-\infty<x<\infty}$')
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig('Lab09_Q01_c_norm.png')
plt.close()

print('Normalization calculated and plotted.')
print(f'Greatest absolute error: {np.format_float_scientific(np.amax(np.abs(normalization - 1)), 3)}')

# Check energy conservation
print()
print('Energy --------------------------------------------------------------------------------------------------------')
energy = np.empty(time_steps, float)
for t in range(time_steps):
    if (t + 1) % 600 == 0:
        print(f'Calculating energy at time step {t + 1} of {time_steps}')
    phi_t = phi[t, :]
    operated_phi_t = np.pad(np.matmul(H_D, phi_t[1:-1]), (1,), 'constant')
    integrands = np.conj(phi_t) * operated_phi_t
    energy[t] = np.real(simpson(integrands, space_axis))

print('Energy calculated.')
print()

plt.figure(figsize=fig_size, dpi=dpi)
plt.plot(time_axis, energy, c='green')

plt.title('Energy over time')
plt.xlabel('Time (s)')
plt.ylabel(r'$E$ (J)')
plt.grid()

plt.tight_layout()
plt.savefig('Lab09_Q01_c_energy.png')
plt.close()

e_range = np.amax(energy) - np.amin(energy)
print(f'Range: {np.format_float_scientific(e_range, 3)} J')
print(f'STD (ddof=1): {np.format_float_scientific(np.std(energy, ddof=1), 3)} J')
print(f'Mean: {np.format_float_scientific(np.mean(energy), 3)} J')
