"""
PHY 407: Computational Physics, Lab 01, Question 2
Author: Amara Winston, John Wood
Date: September 15, 2022

This script modifies that from question 1 to include the gravitational force of Jupiter while calculating the orbit
of the Earth. Different masses for the Jupiter are considered. An asteroid is considered by changing the mass and
initial position of the Earth. We assume that the Sun is not affected by the Earth or Jupiter, and that Jupiter is
ont affected by the Earth. Therefore, in all cases, we calculate the orbit of Jupiter the use its position to
calculate its gravitational effect on the Earth.

Outputs:
    - Lab01_Q2_trajectory.png (trajectories of Earth and Jupiter)
    - Lab01_Q2_trajectory_massive.png (trajectories of Earth and Jupiter of one solar mass)
    - Lab01_Q2_trajectory_massive_escape.png (trajectories of Earth and Jupiter of one solar mass with escape of Earth)
    - Lab01_Q2_trajectory_asteroid.png (trajectories of Jupiter and asteroid)

"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# CONSTANTS ============================================================================================================
# (distances in AU, time in yr, speeds in AU/yr, mass in solar masses)
G = 39.5 # AU^3 Ms^{-1} yr^{-2}
Ms = 1.0 # mass of the Sun


# NUMERICAL INTEGRATION ================================================================================================
def numerically_integrate_gravity_jupiter(t_0, t_final, delta_t, x_e_0, y_e_0, v_x_e_0, v_y_e_0, x_j_0, y_j_0,
                                          v_x_j_0, v_y_j_0, m_j):
    # Create arrays
    t = np.arange(t_0, t_final + delta_t, delta_t)  # add delta_t to t_final so integration includes final time step
    n = len(t)  # number of time steps
    x_j = np.empty(n)
    y_j = np.empty(n)
    v_x_j = np.empty(n)
    v_y_j = np.empty(n)
    x_e = np.empty(n)
    y_e = np.empty(n)
    v_x_e = np.empty(n)
    v_y_e = np.empty(n)

    # Set initial conditions
    x_j[0] = x_j_0
    y_j[0] = y_j_0
    v_x_j[0] = v_x_j_0
    v_y_j[0] = v_y_j_0
    x_e[0] = x_e_0
    y_e[0] = y_e_0
    v_x_e[0] = v_x_e_0
    v_y_e[0] = v_y_e_0

    # Calculate Jupiter's orbit (which is is not affected by the Earth)
    for i in range(n - 1):
        r = sqrt(x_j[i] ** 2 + y_j[i] ** 2)

        v_x_j[i + 1] = v_x_j[i] - ((G * Ms * x_j[i] * delta_t) / (r ** 3))
        v_y_j[i + 1] = v_y_j[i] - ((G * Ms * y_j[i] * delta_t) / (r ** 3))

        x_j[i + 1] = x_j[i] + (v_x_j[i + 1] * delta_t)
        y_j[i + 1] = y_j[i] + (v_y_j[i + 1] * delta_t)

    # Calculate Earth's orbit (affected by Jupiter)
    for i in range(n - 1):
        r_e_s = sqrt(x_e[i] ** 2 + y_e[i] ** 2) # Earth to Sun distance
        x_e_j = x_e[i] - x_j[i] # x distance from Earth to Jupiter
        y_e_j = y_e[i] - y_j[i] # y distance from Earth to Jupiter
        r_e_j = sqrt(x_e_j ** 2 + y_e_j ** 2) # Earth to Jupiter distance

        v_x_e[i + 1] = v_x_e[i] - ((G * Ms * x_e[i] * delta_t) / (r_e_s ** 3)) \
                                - ((G * m_j * x_e_j * delta_t) / (r_e_j ** 3))
        v_y_e[i + 1] = v_y_e[i] - ((G * Ms * y_e[i] * delta_t) / (r_e_s ** 3)) \
                                - ((G * m_j * y_e_j * delta_t) / (r_e_j ** 3))

        x_e[i + 1] = x_e[i] + (v_x_e[i + 1] * delta_t)
        y_e[i + 1] = y_e[i] + (v_y_e[i + 1] * delta_t)

    return t, x_e, y_e, v_x_e, v_y_e, x_j, y_j, v_x_j, v_y_j


# ACTUAL JUPITER MASS ==================================================================================================
results = numerically_integrate_gravity_jupiter(t_0=0, t_final=10, delta_t=0.0001, x_e_0=1.0, y_e_0=0.0, v_x_e_0=0.0,
                                                v_y_e_0=6.18, x_j_0=5.2, y_j_0=0.0, v_x_j_0=0.0, v_y_j_0=2.63,
                                                m_j=0.001)
t, x_e, y_e, v_x_e, v_y_e, x_j, y_j, v_x_j, v_y_j = results

plt.figure(0, figsize=(8, 5), dpi=150)
plt.plot(x_e, y_e, color='C0', lw=1, label='Earth')
plt.plot(x_j, y_j, color='C1', lw=1, label='Jupiter')
plt.plot(0, 0, 'ro', markersize=2, label='Sun')
plt.axis('equal') # make axis have equal scaling
plt.title('Trajectory of Earth and Jupiter')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.legend()
plt.savefig('Lab01_Q2_trajectory.png')

# MORE MASSIVE JUPITER =================================================================================================
results = numerically_integrate_gravity_jupiter(t_0=0, t_final=3, delta_t=0.0001, x_e_0=1.0, y_e_0=0.0, v_x_e_0=0.0,
                                                v_y_e_0=6.18, x_j_0=5.2, y_j_0=0.0, v_x_j_0=0.0, v_y_j_0=2.63,
                                                m_j=1)
t, x_e, y_e, v_x_e, v_y_e, x_j, y_j, v_x_j, v_y_j = results

plt.figure(1, figsize=(8, 5), dpi=150)
plt.plot(x_e, y_e, color='C0', lw=1, label='Earth')
plt.plot(x_j, y_j, color='C1', lw=1, label='Jupiter')
plt.plot(0, 0, 'ro', markersize=2, label='Sun')
plt.axis('equal') # make axis have equal scaling
plt.title('Trajectory of Earth and More Massive Jupiter')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.legend()
plt.savefig('Lab01_Q2_trajectory_massive.png')

# EARTH ESCAPE =========================================================================================================
results = numerically_integrate_gravity_jupiter(t_0=0, t_final=3.75, delta_t=0.0001, x_e_0=1.0, y_e_0=0.0, v_x_e_0=0.0,
                                                v_y_e_0=6.18, x_j_0=5.2, y_j_0=0.0, v_x_j_0=0.0, v_y_j_0=2.63,
                                                m_j=1)
t, x_e, y_e, v_x_e, v_y_e, x_j, y_j, v_x_j, v_y_j = results

plt.figure(2, figsize=(8, 5), dpi=150)
plt.plot(x_e, y_e, color='C0', lw=1, label='Earth')
plt.plot(x_j, y_j, color='C1', lw=1, label='Jupiter')
plt.plot(0, 0, 'ro', markersize=2, label='Sun')
plt.axis('equal') # make axis have equal scaling
plt.title('Trajectory of Earth and More Massive Jupiter')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.legend()
plt.savefig('Lab01_Q2_trajectory_massive_escape.png')

# ASTEROID ORBIT =======================================================================================================
results = numerically_integrate_gravity_jupiter(t_0=0, t_final=20, delta_t=0.0001, x_e_0=3.3, y_e_0=0.0, v_x_e_0=0.0,
                                                v_y_e_0=3.46, x_j_0=5.2, y_j_0=0.0, v_x_j_0=0.0, v_y_j_0=2.63,
                                                m_j=0.001)
t, x_e, y_e, v_x_e, v_y_e, x_j, y_j, v_x_j, v_y_j = results

plt.figure(3, figsize=(8, 5), dpi=150)
plt.plot(x_e, y_e, color='C0', lw=1, label='Asteroid')
plt.plot(x_j, y_j, color='C1', lw=1, label='Jupiter')
plt.plot(0, 0, 'ro', markersize=2, label='Sun')
plt.axis('equal') # make axis have equal scaling
plt.title('Trajectory of Asteroid and Jupiter')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.legend()
plt.savefig('Lab01_Q2_trajectory_asteroid.png')
