"""
PHY 407: Computational Physics, Lab 06, Question 02
Author: John Wood
Date: October 25, 2022

Uses the Verlet algorithm to plot the trajectories of a 16 particle system under only the Lennard-Jones potential.
Total energy over time steps is also found and plotted. The relative and maximum difference in total energy is
printed to check accuracy.

Outputs:
    - Lab06_Q02_a.png: Plot of particle trajectories
    - Lab06_Q02_b.png: Plot of total energy ove time step
    - Printed: Relative and maximum difference of total energy

"""

import numpy as np
import matplotlib.pyplot as plt

# ============ PSEUDOCODE ==============================================================================================
#
# Set plotting preferences
# Set particle count, step count, and box dimensions
#
# ------------ Question 2A ---------------------------------------------------------------------------------------------
# Define function to calculate force (and thus acceleration) for given particle positions
#     Same function as from Question 1
#
# Define function to calculate total force on particle
#     Set total force as 0
#     For each particle other than the one of interest:
#         Add force due to this particle (see previous function) to total force
#     Return total force
#
# Define function to numerically integrate particles
#     Set dt, step count, and initial positions and velocities
#     Set empty arrays for position and velocity over steps, particles and coordinates
#     Set initial positions and velocities
#
#     For each i such that  1 <= i < step count:
#         Follow Verlet method to update position and velocity
#
#     Return position and velocity arrays
#
# Calculate initial conditions for square of 16 particles
# Feed these into function defined above
# Plot position on xy plane
#
# ------------ Question 2B ---------------------------------------------------------------------------------------------
# Define function to find potential energy for given displacement
#     Implement equation 1 from lab document with 1/2 factor (for each particle)
#
# Set empty potential energy array for each particle over time steps
# Calculate potential energy for each particle and time step using function from above
# Calculate kinetic energy for each particle and time step using velocity array from before
# Sum potential and kinetic energy for each particle to get total energy over time steps
#
# Plot total energy over time step
#
# Calculate and print relative difference between minimum and maximum total energy
#
# ======================================================================================================================

# ============ CONSTANTS ===============================================================================================
# Plotting preferences
fig_size_1, fig_size_2 = (10, 10), (10, 8)
dpi = 300
plt.rcParams.update({'font.size': 15})

# Constants
particle_count = 16 # must be a square number given how initial conditions are defined
step_count = 1000
box_x = 4
box_y = 4

epsilon = 1
sigma = 1

# ============ QUESTION 2A =============================================================================================
print('============ QUESTION 2A ======================================================================================')


def force(r) -> np.array([float, float]):
    """
    Finds force given some displacement vector (delta x, delta y).
    """
    return 24 * r * (((r[0] ** 2 + r[1] ** 2) ** (-4)) - (2 * (r[0] ** 2 + r[1] ** 2) ** (-7)))


def total_force(position, time_index, particle_index):
    """
    Find total force on particle at particle_index at time_index given position array of all particles.
    """
    force_sum = np.zeros(2, float)
    for i in range(particle_count):
        if i != particle_index:
            r = position[time_index, i, :] - position[time_index, particle_index, :]
            force_sum += force(r)
    return force_sum


def integrate_particles(dt, steps, initial_position, initial_velocity):
    """
    Uses the Verlet algorithm to integrate particle position and velocity over time steps.
    """
    # As x/y coordinates. Axes are (time step, particle, coordinate)
    position = np.empty((steps, particle_count, 2), float)
    velocity = np.empty((steps * 2, particle_count, 2), float)

    # Set initial position and velocity
    position[0, :, :] = initial_position
    velocity[0, :, :] = initial_velocity

    # Loop through time steps, skipping the first which is initial conditions
    for step in range(1, steps):
        if step % 200 == 0:
            print(f'At step {step} of {steps}...')

        # Indices (note that velocity arrays stores values for each half step)
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


# Calculate initial conditions
dx = box_x/np.sqrt(particle_count)
dy = box_y/np.sqrt(particle_count)

x_grid = np.arange(dx/2, box_x, dx)
y_grid = np.arange(dy/2, box_y, dy)

xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
x_initial = xx_grid.flatten()
y_initial = yy_grid.flatten()

position_0 = np.column_stack((x_initial, y_initial))
velocity_0 = np.zeros((particle_count, 2), float)

# Get integration
position, velocity = integrate_particles(0.01, step_count, position_0, velocity_0)

# Plotting
print('Plotting...')
plt.figure(figsize=fig_size_1, dpi=dpi)
plt.title(f'Trajectories for {particle_count} Particles under only the Lennard-Jones Potential')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()

for particle in range(particle_count):
    plt.plot(position[:, particle, 0], position[:, particle, 1], '.')

plt.savefig('Lab06_Q02_a.png')

# ============ QUESTION 2B =============================================================================================
print()
print('============ QUESTION 2B ======================================================================================')


def potential_energy(r):
    """
    Calculates the potential energy for a single particle due to another given a displacement vector (delta x, delta y).
    """
    r_mag = np.sqrt(np.sum(np.square(r)))
    return 2 * epsilon * ((sigma / r_mag) ** 12 - (sigma / r_mag) ** 6)


# Empty array for potential for each particle, and over time steps
potential = np.empty((step_count, particle_count), float)

# Calculate potential energy
print('Calculating potential energy...')
for step in range(step_count):
    for particle in range(particle_count):
        total_potential_energy = 0.0

        for i in range(particle_count):
            if i != particle:
                r = position[step, i, :] - position[step, particle, :]
                total_potential_energy += potential_energy(r)

        potential[step, particle] = total_potential_energy

# Calculate kinetic energy
print('Calculating kinetic energy...')
# Only consider every other velocity (since they were found in half steps)
velocity = velocity[::2, :, :]

# Find magnitude of velocity
velocity_squared = np.sum(np.square(velocity), axis=2)
kinetic_energy = 0.5 * velocity_squared # because mass = 1

# Find kinetic and potential energy for each time step (sum over particles for each time step)
potential = np.sum(potential, axis=1)
kinetic_energy = np.sum(kinetic_energy, axis=1)

total_energy = potential + kinetic_energy

# Plotting
print('Plotting...')
plt.figure(figsize=fig_size_2, dpi=dpi)
plt.title(f'Total Energy for {particle_count} Particle System')
plt.xlabel('Step')
plt.ylabel('Total Energy')
plt.grid()

plt.plot(total_energy)

plt.savefig('Lab06_Q02_b.png')

# Statistics
print()
print('TOTAL ENERGY STATISTICS ---------------------------------------------------------------------------------------')

e_min = np.min(total_energy)
e_max = np.max(total_energy)

print(f'Energy conserved within {np.format_float_scientific(((e_max - e_min) / abs(e_min)) * 100, precision=3)} %')
