"""
PHY 407: Computational Physics, Lab 10, Question 01
Author: John Wood
Date: November 29, 2022

Simulates both Brownian motion and Diffusion-Limited Aggregation. Both situations take place on a grid_dim by
grid_dim lattice. At each time step, particles move either one cell up, down, left, or right. (x, y) coordinate pairs
are written as (i, j), while time is marked as t.

Outputs:
    - Lab10_Q01_a_ivt.png: i v.s. t for Brownian motion
    - Lab10_Q01_a_jvt.png: j v.s. t for Brownian motion
    - Lab10_Q01_a_jvi.png: j v.s. i for Brownian motion (the particle's trajectory)
    - Lab10_Q01_b.png: Illustration of final particle positions for diffusion-limited aggregation

"""

# ============ PSEUDOCODE ==============================================================================================
# Set plotting preferences
#
# NOTE: Here, a 'move' is a movement in some direction
# Define function to apply move to a position:
#     Set current x and y position
#     Set move code
#     If move code is that of UP (0):
#         return x and y + 1
#     and so on for DOWN, LEFT, and RIGHT (1, 2, and 3)
#
# ------------ Question 1A ---------------------------------------------------------------------------------------------
# Set number of time steps, and grid size L
# Define results array with dimensions [time steps + 1, 2 (i/j)] (+1 for initial conditions)
# Set initial conditions in results array
# Define UP, DOWN, LEFT, and RIGHT constants as 0, 1, 2, and 3 respectively
# Set current i and current j as the centre of the grid
# For each time step:
#   Shuffle the list [0, 1, 2, 3]
#   For each element in the shuffled list:
#       Get new position from chosen element, current position, and function from above
#       If new position is not off the grid:
#           Set current position as new position and store new position in results array
#           Break from the for loop of elements in the shuffled lise
#
# Plot i vs time steps
# Plot j vs time steps
# Plot j vs i
#
# ------------ Question 1B ---------------------------------------------------------------------------------------------
# NOTE: here, an anchored square is one that is "occupied by an anchored particle" (Newman, pg. 500)
# Keep definitions of grid size and direction integers from above
# Define LxL grid of booleans (really just 1 or 0 integers) to mark anchored squares (from DLA-start.py)
#
# Define stick function which returns whether a particle at a given position is to be anchored and anchors it
#     Set the current position
#     If any adjacent squares are off the grid (paraphrase from Newman, pg. 500):
#         Set current position as anchored
#         Return True
#     If any adjacent squares are anchored:
#         Set current position as anchored
#         Return True
#     All checks have passed, so return False
#
# while value of anchor points array is 0 at centre point:
#     Set current i and j position as the centre of the grid
#
#     while current position is not stuck (check by passing current position to stuck function)
#     Choose a random integer in [0, 3]
#     Get new position from chosen element, current position, and function from above
#     Set current position as new position
#
# Plot anchored particles
#
# ======================================================================================================================

import random
import numpy as np
import matplotlib.pyplot as plt

# ============ SETUP ===================================================================================================
print('============ SETUP ============================================================================================')
# Plotting preferences
fig_size_base = (6.4, 4.8)
fig_size_grid = (6.4, 6.4)
dpi = 300
plt.rcParams.update({'font.size': 13})

# Problem constants (which are the same for both sub questions)
grid_dim = 101 # This is L
UP, DOWN, LEFT, RIGHT = range(4)


# Functions used for both sub questions
def apply_move(move_code, x, y):
    if move_code == UP:
        return x, y + 1
    elif move_code == DOWN:
        return x, y - 1
    elif move_code == LEFT:
        return x - 1, y
    else:
        return x + 1, y


print('Done\n')

# ============ QUESTION 1A =============================================================================================
print('============ QUESTION 1A ======================================================================================')
# Set parameters and results
time_steps = 5_000
results = np.empty((time_steps + 1, 2), int)

# Set initial conditions
current_i = (grid_dim - 1) // 2
current_j = current_i
results[0, :] = current_i, current_j

# Update position
for t in range(1, time_steps + 1):
    if t % 1000 == 0:
        print(f'Finding position at time {t} of {time_steps}')
    for random_move_code in random.sample(range(4), 4):
        # Apply chosen move
        new_i, new_j = apply_move(random_move_code, current_i, current_j)

        # Check if new position is valid
        if (0 <= new_i < grid_dim) and (0 <= new_j < grid_dim):
            current_i, current_j = new_i, new_j
            results[t, :] = current_i, current_j
            break

# Plotting
# i vs t
plt.figure(figsize=fig_size_base, dpi=dpi)
plt.plot(results[:, 0], c='blue')
plt.gca().set_ylim(0, grid_dim - 1)
plt.gca().set_xlim(0, time_steps)
plt.title(r'Simulation of Brownian Motion: $i$ v.s. $t$')
plt.xlabel(r'$t$ (stesps)')
plt.ylabel(r'$i$')
plt.grid()
plt.tight_layout()
plt.savefig('Lab10_Q01_a_ivt.png')
plt.close()

# j vs t
plt.figure(figsize=fig_size_base, dpi=dpi)
plt.plot(results[:, 1], c='red')
plt.gca().set_ylim(0, grid_dim - 1)
plt.gca().set_xlim(0, time_steps)
plt.title(r'Simulation of Brownian Motion: $j$ v.s. $t$')
plt.xlabel(r'$t$ (steps)')
plt.ylabel(r'$j$')
plt.grid()
plt.tight_layout()
plt.savefig('Lab10_Q01_a_jvt.png')
plt.close()

# j vs i
plt.figure(figsize=fig_size_grid, dpi=dpi)
plt.plot(results[:, 0], results[:, 1], c='green')
plt.plot(results[0, 0], results[0, 1], c='red', marker='o', markersize=6) # plot start point
plt.plot(results[-1, 0], results[-1, 1], c='red', marker='o', markersize=6) # plot end point
plt.gca().set_ylim(0, grid_dim - 1)
plt.gca().set_xlim(0, grid_dim - 1)
plt.title(r'Simulation of Brownian Motion: $j$ v.s. $i$')
plt.xlabel(r'$i$')
plt.ylabel(r'$j$')
plt.grid()
plt.tight_layout()
plt.savefig('Lab10_Q01_a_jvi.png')
plt.close()

print('Done.\n')

# ============ QUESTION 1B =============================================================================================
print('============ QUESTION 1B ======================================================================================')
anchored = np.zeros((grid_dim, grid_dim), int)
centre_coordinate = (grid_dim - 1) // 2


def stick(x, y):
    if x == 1 or x == grid_dim - 2 or y == 1 or y == grid_dim - 2:
        # Position is on boundary
        anchored[x, y] = 1
        return True
    if anchored[x - 1, y] or anchored[x + 1, y] or anchored[x, y -1] or anchored[x, y + 1]:
        # Position is touching an anchored point
        anchored[x, y] = 1
        return True
    return False


particle_count = 0
while not anchored[centre_coordinate, centre_coordinate]: # keep adding particles until centre is anchored
    particle_count += 1
    if particle_count % 400 == 0:
        print(f'Particle {particle_count} is walking')
    current_i, current_j = centre_coordinate, centre_coordinate

    # Start walk
    while not stick(current_i, current_j):
        current_i, current_j = apply_move(random.randint(0, 3), current_i, current_j)

print(f'Centre particle is anchored. Took {particle_count} particle.')

# Plot anchored point
plt.figure(figsize=fig_size_grid, dpi=dpi)
plt.imshow(np.rot90(anchored), cmap='binary')
plt.gca().set_ylim(0, grid_dim - 1)
plt.gca().set_xlim(0, grid_dim - 1)
plt.title('Diffusion-Limited Aggregation')
plt.xlabel(r'$i$')
plt.ylabel(r'$j$')
plt.grid()
plt.tight_layout()
plt.savefig('Lab10_Q01_b.png')
plt.close()
