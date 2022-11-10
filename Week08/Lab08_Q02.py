"""
PHY 407: Computational Physics, Lab 07, Question 01
Author: John Wood
Date: November 1, 2022

Plots the trajectory of a ball bering orbiting a rod, as calculated by an adaptive and non-adaptive RK4 algorithm (
wrt step size). Each algorithm is times. For the adaptive algorithm, step size is plotted over time, and also with
velocity.

Outputs:
    - LAB07_Q01_a.png: Trajectories according to adaptive and non-adaptive RK4 algorithms
    - Printed: time taken for each algorithm
    - LAB07_Q01_c.png: Step size over time
    - LAB07_Q01_c2.png: Step size and velocity over time

"""

import numpy as np
import scipy.special as sps
from math import sqrt, pi
import matplotlib.pyplot as plt

# ============ CONSTANTS ===============================================================================================
# Plotting preferences
fig_size = (8, 12)
plt.rcParams.update({'font.size': 15})

# Problem constants
x_0 = 0 # m
x_final = 1 # m
L = x_final # m (x_final = L)
J_x = 500
dx = L / J_x # m

t_0 = 0 # s
t_final = 4 # s
J_t = 400
dt = t_final / J_t # s

g = 9.81 # ms^{-2}
H = 0.01 # m


def topography(x):
    return 0 # m


# Initial conditions
u_t_0 = 0 # ms^{-1} (fluid's x velocity at initial time, for all x)

A = 0.002 # m
mu = 0.5 # m
sigma = 0.05 # m
eta_bar = 0.5 * sqrt(pi) * A * sigma * (sps.erf(mu / sigma) + sps.erf((L - mu) / sigma)) / L


def eta_t_0(x):
    return H + 0.002 * np.exp(-1 * ((x - 0.5) ** 2) / (0.05 ** 2)) - eta_bar


# ============ QUESTION 2B =============================================================================================
# Create x-axis
x_axis = np.linspace(x_0, x_final, J_x)

# Create empty results arrays. Axes are, respectively, position and time
u = np.empty((J_x, J_t), float)
eta = np.empty((J_x, J_t), float)

# Set initial conditions
u[:, 0] = 0.
eta[:, 0] = eta_t_0(x_axis)

plt.plot(x_axis, eta[:, 0])
plt.show()

