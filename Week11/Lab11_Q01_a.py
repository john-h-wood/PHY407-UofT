"""
PHY 407: Computational Physics, Lab 11, Question 01 a
Author: John Wood
Date: December 6, 2022

*** NOTE ***
This code is adapted from salesman.py, the solution to Example 10.4 from Newman.

This script uses simulated annealing to optimize solutions to the travelling salesman problem. Different cooling
rates are used, and their effect on the total distance is investigated. For each cooling rate, three seeds are
tested, so that the optimization goes through three trials. Plots of the final tours are generated.

"""

# ============ PSEUDOCODE ==============================================================================================
# Import required modules (numpy, math functions, pyplot)
# Set plotting preferences (figure size, dpi, font size)
# Define problem constants (number of cities N, Tmax, Tmin)
# Define list of tau's to use
# Define number of trials to do per tau
# Define list of seeds for trials (must have length (len(taus)*trials_per_tau))

# Define function to compute magnitude of vector:
#   Set vector (array) v
#   Return the sum of the squares of the components of v
#
# Define function to compute the total distance of a tour (tour = closed path passing all cities)
#   Set tour (ordered array of city coordinates)
#   Define s=0
#   For each i in range(0, number_of_cities):
#       Find take vector difference tour[i+1] - tour[i]
#       Add the magnitude of that difference to s
#   Return s
#
# Define function which plots a list of tours in subplots
# Define function to set cities:
#     Set seed for number generation to a pre-determined value
#     Create empty array to store (number_of_cities  + 1) pairs of coordinates
#     For each i in range(0, number_of_cities):
#         Set tour[i] to a pair of random numbers
#     Set tour[number_of_cities] to tour[0] (so that tour is closed)
#     Return default tour from city 0 to city 1, city 1 to city 2, and so on
#
# For each tau:
#     define empty list of tours
#     define empty list of final distances
#     For each i in range(0, tries_per_tau):
#         Initialize the cities and default tour using function above
#         Set the seed to the appropriate predefined value the list from before
#         Set T to max_temp
#
#         while T > min_temp:
#             Exponentially decay T according to tau
#             choose random cities to flip in the tour
#             determine the difference in difference this flip causes
#             If this energy difference satisfies the acceptance condition:
#                 do nothing
#             Otherwise:
#                 Undo the flip
#         append final tour to list of tours
#         append final distance to list of final distances
#     plot the final tours for this tau using function from above, including final distances in titles
#     print the mean final distance for this tau
#
# ======================================================================================================================

import numpy as np
import random as rand
from math import exp
import matplotlib.pyplot as plt

# ============ CONSTANTS ===============================================================================================
# Plotting preferences
fig_size_single = (6.4, 4.8)
fig_size_multi = (6.4, 6)
dpi = 300
plt.rcParams.update({'font.size': 13})

# Problem constants
number_of_cities = 25
max_temp = 10.0
min_temp = 1e-3

taus = (1e2, 1e3, 1e4, 2e4, 1e5)
tries_per_tau = 3
seeds_for_taus = ((10, 20, 33), (40, 50, 60), (70, 80, 90), (100, 110, 120), (130, 140, 150))

# ============ FOR MARKER ==============================================================================================
# Set this boolean to True to reduce the number of trials, and thus decrease runtime
reduce_trials = False

if reduce_trials:
    taus = (1e4, 2e4)
    tries_per_tau = 2
    seeds_for_taus = ((10, 20), (30, 40))


# ============ USEFUL FUNCTIONS ========================================================================================
def magnitude(_v):
    """
    Calculate the magnitude of a vector.

    Args:
        _v: The vector.

    Returns:
        The magnitude.

    """
    return np.sqrt(np.sum(np.square(_v)))


def total_distance(_tour):
    """
    Calculate the total distance of a tour.

    Args:
        _tour: The tour, a closed array of 2d vectors. The first dimension of this array is the point.

    Returns:
        The total distance.

    """
    d = float()
    for i in range(number_of_cities):
        d += magnitude(_tour[i+1] - _tour[i])
    return d


def plot_single_tour(_ax, _tour, _title):
    """
    Plot a single tour on a given matplotlib axis.

    Args:
        _ax: The matplotlib axis.
        _tour: The tour.
        _title: The title for the plot.

    Returns:
        None.

    """

    for i in range(number_of_cities):
        _ax.plot((_tour[i][0], _tour[i + 1][0]), (_tour[i][1], _tour[i + 1][1]), ls='-', c='black')
        _ax.plot(*_tour[i], marker='o', c='blue', markersize=5)
    # Style preference: draw last/first city again to make city markers above travel lines
    _ax.plot(*_tour[-1], marker='o', c='blue', markersize=5)

    _ax.set_title(_title)
    _ax.set_xlabel(r'$x$')
    _ax.set_ylabel(r'$y$')


def plot_tours(_tours, _suptitle, _titles, _save_title):
    """
    Plots multiple tours in one figure.
    Args:
        _tours: The list of tours to plot.
        _suptitle: The suptitle for the figure.
        _titles: The list of titles for the tours.
        _save_title: The filename to save the figure to.

    Returns:
        None.

    """
    if len(_tours) == 1:
        raise ValueError('The number of tours given should be greater than one.')

    fig, axes = plt.subplots(len(_tours), 1, sharex=True, sharey=True, figsize=fig_size_multi, dpi=dpi)

    for ax, tour, title in zip(axes, _tours, _titles):
        plot_single_tour(ax, tour, title)

    fig.suptitle(_suptitle)
    plt.tight_layout()
    plt.savefig(_save_title)
    plt.close()


# ============ PROBLEM SETUP ===========================================================================================
def set_cities():
    """
    Set the cities at their initial positions.

    Returns:
        The default tour

    """
    # Set city positions
    rand.seed(314159265358) # makes pretty nicely spaced points, and is also the first few digits of pi!
    tour = np.empty((number_of_cities + 1, 2), float)
    for i in range(number_of_cities):
        tour[i] = rand.random(), rand.random()
    tour[number_of_cities] = tour[0] # make tour closed

    return tour


# uncomment these lines below to view base tour (useful to choose the seed)
# --------------------------------------------------------------------------------
# rand.seed(314159265358)
#
# tour = np.empty((number_of_cities + 1, 2), float)
# for i in range(number_of_cities):
#     tour[i] = rand.random(), rand.random()
# tour[number_of_cities] = tour[0]
#
# plt.figure(figsize=fig_size_single, dpi=dpi)
# plot_single_tour(plt.gca(), tour, 'Base tour (to test seed)')
# plt.savefig('base_tour.png')
# plt.close()
# --------------------------------------------------------------------------------

# ============ SIMULATED ANNEALING =====================================================================================
print('========= SIMULATED ANNEALING =================================================================================')
print('Please note: this process takes a while (about 12 minutes on my computer), especially for the last and large '
      'tau.\nIf the marker desires, they may set the boolean reduce_trials to True. This will reduce the number of '
      'trials,\ndecreasing runtime.\n')

for tau_idx, tau in enumerate(taus):
    print(f'Working on tau {tau_idx + 1} of {len(taus)}')
    final_tours = list()
    final_distances = list()

    # For each tau, we go through its trial seeds, do simulated annealing, and store final tour and distance
    for seed_idx, try_seed in enumerate(seeds_for_taus[tau_idx]):
        print(f'        Working on seed {seed_idx + 1} of {len(seeds_for_taus[tau_idx])}')
        # Initialize cities, default tour, and working distance
        tour = set_cities()
        working_distance = total_distance(tour)

        # Initialize temperature variables
        try_temp = max_temp
        try_t = 0

        # Initialize seed
        rand.seed(try_seed)

        # Annealing
        while try_temp > min_temp:

            # Cooling
            try_t += 1
            try_temp = max_temp * exp(-try_t / tau)

            # Choose two cities to swap and make sure they are distinct
            i, j = rand.randrange(1, number_of_cities), rand.randrange(1, number_of_cities)
            while i == j:
                i, j = rand.randrange(1, number_of_cities), rand.randrange(1, number_of_cities)

            # Swap them and calculate the change in distance
            oldD = working_distance
            tour[i, 0], tour[j, 0] = tour[j, 0], tour[i, 0]
            tour[i, 1], tour[j, 1] = tour[j, 1], tour[i, 1]
            working_distance = total_distance(tour)
            deltaD = working_distance - oldD

            # If the move is rejected, swap them back again
            if rand.random() > exp(-deltaD / try_temp):
                tour[i, 0], tour[j, 0] = tour[j, 0], tour[i, 0]
                tour[i, 1], tour[j, 1] = tour[j, 1], tour[i, 1]
                working_distance = oldD

        # Save final tour and distance
        final_tours.append(tour)
        final_distances.append(working_distance)

    # Plot results for this tau
    suptitle = rf'$\tau={np.format_float_scientific(tau, precision=3)}$'
    titles = [f's={s}, D={np.format_float_scientific(d, precision=3)}' for s, d in zip(seeds_for_taus[tau_idx],
                                                                                       final_distances)]

    plot_tours(final_tours, suptitle, titles, f'Lab11_Q01_a_tau{tau_idx}.png')
    mean_distance = np.format_float_scientific(np.mean(final_distances), precision=3)
    print(f'Average distance for tau {tau_idx + 1}: {mean_distance}')












