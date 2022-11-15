"""
PHY 407: Computational Physics, Lab 08, Question 02
Author: John Wood
Date: November 14, 2022

Uses the FTCS algorithm to solve the 1D shallow water equations. "A small Gaussian peak" (lab handout) is the initial
conditions. Plots for t=0, 1, and 4 seconds are saved.

Outputs:
    - Lab08_Q02_b_0s.png: Wave at t=0s
    - Lab08_Q02_b_1s.png: Wave at t=1s
    - Lab08_Q02_b_4s.png: Wave at t=4s

"""

# ============ PSEUDOCODE ==============================================================================================
# Define plotting preferences
# Define problem constants and initial conditions
#   Use scipy's error function to calculate value of mean in Eq. 8 of handout
#   With above, define function that calculates eta_b at t=0s for a given x-axis
#
# ------------ QUESTION 2B ---------------------------------------------------------------------------------------------
# Define empty results array with dimensions ([u/eta], position steps, time steps)
# Set initial conditions in results array
#
# Define function to implement Eq. 7 from handout (equation for F):
#   Set position and time index
#   Get u and eta at above indices
#   Calculate and return F(u, eta)
#
# For each time step in results array, except the last:
#   For each position step in results array, including the last:
#       Find (u, eta) at the current position and *next* time step:
#           If position == 0 or is the last position ( == J), use Eq. 9 from handout
#           Otherwise, use Eq. 1 and 2 from submitted lab document (ie the document that we wrote, not the handout)
#
# Plot eta over x, H and eta_b at t=0, 1, and 4s.
#
# ======================================================================================================================

import numpy as np
import scipy.special as sps
from math import pi, sqrt
import matplotlib.pyplot as plt

# ============ CONSTANTS ===============================================================================================
print('============ SETUP ============================================================================================')

# Plotting preferences
fig_size = (6.4, 4.8)
dpi = 300
plt.rcParams.update({'font.size': 13})
plotting_times = (0, 1, 4) # s

# Problem constants
g = 9.81 # ms^{-2}
H = 0.01 # m
eta_b = 0

x_0 = 0 # m. Note: the calculation of eta_bar relies on this being zero!
x_final = 1 # m
J_x = 50
dx = (x_final - x_0) / J_x # m

t_0 = 0 # s
t_final = 4 # s
J_t = 400
dt = (t_final - t_0) / J_t # s

# Initial conditions
u_t_0 = 0.

A = 0.002 # m
mu = 0.5 # m
sigma = 0.05 # m
print('Calculating eta_bar...')
eta_bar = (1 / (2 * x_final)) * sqrt(pi) * A * sigma * (sps.erf(mu / sigma) + sps.erf((x_final - mu) / sigma))
print('Done.')


def eta_t_0(x):
    return H + A * np.exp((-1 * (x - mu) ** 2) / (sigma ** 2)) - eta_bar


# ============ QUESTION 2B =============================================================================================
print()
print('============ QUESTION 2B ======================================================================================')
# Initialize empty results array. Dimensions are [u / eta], position, time
results = np.empty((2, J_x + 1, J_t + 1), float) # since we have, for example, x_0, x_1, ..., x_J

# Set initial conditions
x_axis = np.linspace(x_0, x_final, J_x + 1)
results[0, :, 0] = 0.
results[1, :, 0] = eta_t_0(x_axis)

def F(position, time):
    """
    Gives F as defined by Eq. 7 in lab handout for a given position and time index.
    """
    u = results[0, position, time]
    eta = results[1, position, time]
    return np.array((0.5 * (u ** 2) + (g * eta), (eta - eta_b) * u))


# Step through and update results array, updating (time + 1) and (position)
for time in range(J_t):
    for pos in range(J_x + 1):
        # Separate cases for points on spatial boundary
        if pos == 0:
            results[:, 0, time + 1] = results[:, 0, time] - (dt / dx) * (F(1, time) - F(0, time))
        elif pos == J_x:
            results[:, J_x, time + 1] = results[:, J_x, time] - (dt / dx) * (F(J_x, time) - F(J_x - 1, time))

        else:
            results[:, pos, time + 1] = results[:, pos, time] - (dt / (2 * dx)) * (F(pos + 1, time) - F(pos - 1, time))

# Plotting
print('Plotting...')
for time in plotting_times:
    time_index = int((time - t_0) / dt)

    plt.figure(figsize=fig_size, dpi=dpi)
    plt.hlines(H, x_0, x_final, colors='grey', linestyles='dashed', label=r'$H$')
    plt.plot(x_axis, results[1, :, time_index], label=r'$\eta$', lw=2.5)

    plt.legend()
    plt.xlabel(r'$x$ (m)')
    plt.ylabel(r'$z$ (m)')
    plt.title(rf'$\eta$ at $t$={time} s')

    plt.tight_layout()
    plt.savefig(f'Lab08_Q02_b_{time}s.png')

print('Done.')
