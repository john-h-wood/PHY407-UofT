"""
PHY 407: Computational Physics, Lab 05, Question 1
Author: John Wood
Date: October 15, 2022

TODO

Outputs:
    - TODO

"""

import numpy as np
import scipy.constants as spc
import matplotlib.pyplot as plt

# ============ CONSTANTS ===============================================================================================
# Plotting preferences
fig_size = (12, 8)
dpi = 200
plt.rcParams.update({'font.size': 12})

# Numerical constants
m = 1 # kg
k = 12 # N / kg
c = spc.c # m / s^2
pi = spc.pi
x_c = c * (m/k) ** 0.5 # meters

# ============ QUESTION 1A =============================================================================================
initial_displacements = (1, x_c, 10 * x_c)
solutions = list()


# Function for numerical integration of the spring
def numerically_integrate_spring(t_0, t_final, delta_t, x_0, v_0):
    # Create arrays
    t = np.arange(t_0, t_final + delta_t, delta_t)  # add delta_t to t_final so integration includes final time step
    n = len(t)  # number of time steps
    x = np.empty(n, float)
    v = np.empty(n, float)

    # Set initial conditions
    x[0] = x_0
    v[0] = v_0

    for i in range(n - 1):
        v[i + 1] = v[i] + (-k / m) * x[i] * ((1 - ((v[i] ** 2)/(c ** 2))) ** 1.5) * delta_t
        x[i + 1] = x[i] + (v[i + 1] * delta_t)

    return t, x, v


