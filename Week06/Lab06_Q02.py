"""
PHY 407: Computational Physics, Lab 06, Question 02
Author: John Wood
Date: October 23, 2022

TODO

Outputs:
    - TODO

"""

import numpy as np
import matplotlib.pyplot as plt

# ============ CONSTANTS ===============================================================================================
# Plotting preferences
fig_size = (10, 10)
dpi = 300
plt.rcParams.update({'font.size': 15})

# Constants
particle_count = 16
box_x = 4
box_y = 4


# ============ QUESTION 2A =============================================================================================
def force(r) -> np.array([float, float]):
    return 24 * r * (((r[0] ** 2 + r[1] ** 2) ** (-4)) - (2 * (r[0] ** 2 + r[1] ** 2) ** (-7)))


def total_force(position, time_index, particle_index):
    force_sum = np.zeros(2, float)
    for i in range(particle_count):
        if i != particle_index:
            r = position[time_index, i, :] - position[time_index, particle_index, :]
            force_sum += force(r)
    return force_sum


def integrate_particle_position(dt, steps, initial_position, initial_velocity):
    # As x/y coordinates. Axes are (time step, particle, coordinate)
    position = np.empty((steps, particle_count, 2), float)
    velocity = np.empty((steps * 2, particle_count, 2), float)

    # Set initial position and velocity
    position[0, :, :] = initial_position
    velocity[0, :, :] = initial_velocity

    # Loop through time steps, skipping the first which is initial conditions
    for step in range(1, steps):
        if step % 25 == 0:
            print(f'Progress update: at step {step} of {steps - 1}')

        r_t = step - 1
        r_t_plus_dt = step
        v_t_plus_dt = 2 * step
        v_t_plus_half_dt = 2 * step - 1
        v_t_plus_three_half_dt = 2 * step + 1

        # Only first step
        if step == 1:
            for particle in range(particle_count):
                temp_force = total_force(position, r_t, particle)
                velocity[1, particle, :] = velocity[0, particle, :] + (dt / 2) * temp_force

        # Update position and velocity according to Verlet algorithm
        for particle in range(particle_count):
            position[r_t_plus_dt, particle, :] = position[r_t, particle, :] + dt * velocity[v_t_plus_half_dt,
                                                                                            particle, :]

        for particle in range(particle_count):
            temp_force = total_force(position, r_t_plus_dt, particle)
            k = dt * temp_force
            velocity[v_t_plus_dt, particle, :] = velocity[v_t_plus_half_dt, particle, :] + (1/2) * k
            velocity[v_t_plus_three_half_dt, particle, :] = velocity[v_t_plus_half_dt, particle, :] + k

    return position, velocity


dx = box_x/np.sqrt(particle_count)
dy = box_y/np.sqrt(particle_count)
x_grid = np.arange(dx/2, box_x, dx)
y_grid = np.arange(dy/2, box_y, dy)
xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
x_initial = xx_grid.flatten()
y_initial = yy_grid.flatten()

position_0 = np.column_stack((x_initial, y_initial))
velocity_0 = np.zeros((particle_count, 2), float)

position, velocity = integrate_particle_position(0.01, 100, position_0, velocity_0)

# Init plot
plt.figure(figsize=fig_size, dpi=dpi)
plt.title('Trajectories for 16 Particles under only the Lennard-Jones Potential')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()

for particle in range(particle_count):
    plt.plot(position[:, particle, 0], position[:, particle, 1], '.')

plt.savefig('plot.png')

