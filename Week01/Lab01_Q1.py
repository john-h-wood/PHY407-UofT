"""
PHY 407: Computational Physics, Lab 01, Question 1
Author: John Wood
Date: September 15, 2022

This is the script for the pseudocode in question 1b. It approximates the motion of Mercury around the Sun using
numerical integration with first Newtonian then relativistic gravity.

Outputs:
    Newtonian gravity:
        - Lab01_Q1_trajectory_newtonian.png (x vs. y)
        - Lab01_Q1_velocity_newtonian.png (v_x vs. t and v_y vs. t)
        - Difference in angular momenta (vector)
    Relativistic gravity:
        - Lab01_Q1_trajectory_relativistic (x vs. y)

"""
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# CONSTANTS ============================================================================================================
# (distances in AU, time in yr, speeds in AU/yr, mass in solar masses)
G = 39.5 # AU^3 Ms^{-1} yr^{-2}
Ms = 1.0 # mass of the Sun
Mm = 1.65 * (10 ** -7) # mass of mercury


# NUMERICAL INTEGRATION ================================================================================================
def numerically_integrate_gravity(t_0, t_final, delta_t, x_0, y_0, v_x_0, v_y_0, alpha):
    """
    Creates, populates, and returns arrays for time, position, distance and velocity under gravity according to
    initial conditions.

    Used for both Newtonian (pass alpha = 0) and relativistic (pass alpha != 0) gravity.

    :param t_0: Initial time (yr)
    :param t_final: Final time (yr)
    :param delta_t: Time step (yr)
    :param x_0: Initial x position (AU)
    :param y_0: Initial y position (AU)
    :param v_x_0: Initial x velocity (AU/yr)
    :param v_y_0: Initial y velocity (AU/yr)
    :param alpha: Relativistic constant (AU^{-2})
    :return: Time, x, y, distance, v_x, v_y
    """

    # Create arrays
    t = np.arange(t_0, t_final + delta_t, delta_t) # add delta_t to t_final so integration includes final time step
    n = len(t) # number of time steps
    x = np.empty(n)
    y = np.empty(n)
    r = np.empty(n) # store r for later use in calculating angular momentum
    v_x = np.empty(n)
    v_y = np.empty(n)

    # Set initial conditions
    x[0] = x_0
    y[0] = y_0
    v_x[0] = v_x_0
    v_y[0] = v_y_0

    for i in range(n - 1):
        r[i] = sqrt(x[i] ** 2 + y[i] ** 2)

        v_x[i + 1] = v_x[i] - (((G * Ms * x[i] * delta_t) / (r[i] ** 3)) * (1 + (alpha / r[i] ** 2)))
        v_y[i + 1] = v_y[i] - (((G * Ms * y[i] * delta_t) / (r[i] ** 3)) * (1 + (alpha / r[i] ** 2)))

        x[i + 1] = x[i] + (v_x[i + 1] * delta_t)
        y[i + 1] = y[i] + (v_y[i + 1] * delta_t)

    r[-1] = sqrt(x[-1] ** 2 + y[-1] ** 2) # fill last distance

    return t, x, y, r, v_x, v_y


# PYPLOT PARAMS ========================================================================================================
# Use size 12 font
plt.rcParams.update({
    'font.size': 12
})

# NEWTONIAN ============================================================================================================
t, x, y, r, v_x, v_y = numerically_integrate_gravity(t_0=0, t_final=1, delta_t=0.0001, x_0=0.47, y_0=0.0, v_x_0=0.0,
                                                     v_y_0=8.17, alpha=0)
v = np.sqrt(v_x ** 2 + v_y ** 2) # to plot v
# v_x and v_y vs. t
plt.figure(0, figsize=(8, 5), dpi=150)
plt.plot(t, v_x, color='C0', lw=2, label='v_x')
plt.plot(t, v_y, color='C1', lw=2, label='v_y')
plt.plot(t, v, color='C2', lw=2, label='v')
plt.title('Velocity of Mercury over time (Newtonian Gravity))')
plt.xlabel('Time (yr)')
plt.ylabel('Speed (AU/yr)')
plt.legend()
plt.savefig('Lab01_Q1_velocity_newtonian.png')

# x vs. y
plt.figure(1, figsize=(8, 5), dpi=150)
plt.plot(x, y, color='C0', lw=2, label='Position')
plt.plot(0, 0, 'ro', markersize=2, label='Origin')
plt.axis('equal') # make axis have equal scaling
plt.title('Trajectory of Mercury (Newtonian Gravity)')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.legend()
plt.savefig('Lab01_Q1_trajectory_newtonian.png')

# Angular Momentum -----------------------------------------------------------------------------------------------------
# L = mv (cross product) r
# Convert flat arrays to arrays of components for np.cross
v_vectors = np.array([[i_v_x, i_v_y, 0] for i_v_x, i_v_y in zip(v_x, v_y)])
r_vectors = np.array([[i_r_x, i_r_y, 0] for i_r_x, i_r_y in zip(x, y)])

L = np.cross(v_vectors, r_vectors) * Mm # Ms AU^2 yr^{-1}

print(f'Difference in angular momentum: {L[-1] - L[0]}')

# RELATIVISTIC =========================================================================================================
t, x, y, r, v_x, v_y = numerically_integrate_gravity(t_0=0, t_final=1, delta_t=0.0001, x_0=0.47, y_0=0.0, v_x_0=0.0,
                                                     v_y_0=8.17, alpha=0.01)
# x vs. y
plt.figure(2, figsize=(8, 5), dpi=150)
plt.plot(x, y, color='C0', lw=2, label='Position')
plt.plot(0, 0, 'ro', markersize=2, label='Origin')
plt.axis('equal') # make axis have equal scaling
plt.title('Trajectory of Mercury (Relativistic Gravity)')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.legend()
plt.savefig('Lab01_Q1_trajectory_relativistic.png')
