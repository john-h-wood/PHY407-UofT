"""
PHY 407: Computational Physics, Lab 11, Question 01 b and c
Author: John Wood
Date: December 6, 2022

*** NOTE ***
This code is adapted from salesman.py, the solution to Example 10.4 from Newman.

This script uses simulated annealing to optimize solutions to the travelling salesman problem. Different cooling
rates are used, and their effect on the total distance is investigated. For each cooling rate, three seeds are
tested, so that the optimization goes through three trials. Plots of the final tours are generated.

"""

import numpy as np
import random as rand
from math import exp, inf, cos, pi, sqrt
import matplotlib.pyplot as plt

# ============ PSEUDOCODE ==============================================================================================
# Import required modules (numpy, math functions, pyplot)
# Set plotting preferences (figure size, dpi, font size)
# Seed random number generation to pre-determined value
#
# Define function to plot list of x's and y's on a single figure using subplots
#
# Define function to perform simulated annealing:
#    Set the max and min temperature
#    Set tau
#    Set x and y bounds
#    Set 'energy' function f
#    Set starting x, x_0 and starting y, y_0
#    Set current x and current y from x_0 and y_0
#
#    Define empty lists for x and y values
#    Set T to max_temp
#
#    while T > min_temp:
#                 Exponentially decay T according to tau
#                 Calculate candidate (x,y) as sum of current (x, y) and two Gaussian-distributed random numbers
#                 If this new point is outside the x and y bounds:
#                     Append current (non-candidate) x and y to list of x's and y's
#                     Continue to next T
#
#                 Calculate the 'energy' difference associated with this point change
#                 If this energy difference satisfies the acceptance condition:
#                     Set the current (x,y) as the candidate (x,y)
#                     Append current (formerly candidate) (x, y) to list of x's and y's
#    Return list of x's and list of y's
#
# Define function according to Equation 4 from lab handout
# Define pre-determined temperature parameters for this function
# Define x and y bounds and initial (x, y) according to lab handout
# Find list of x's and y's from annealing function
# Print last values from x and y lists
# Plot x's and y's over time steps using function from above
#
# Repeat lines 51-56 but for Equation 5 from lab handout.
#
# ======================================================================================================================

# ============ PREFERENCES =============================================================================================
# Plotting preferences
double_fig_size = (7, 9.6)
dpi = 300
plt.rcParams.update({'font.size': 13})

# FOR MARKER: Set seed to 0 to get the same results from lab report
rand.seed(0)


# ============ SIMULATED ANNEALING =====================================================================================
def anneal(function, x_bounds, y_bounds, x_0, y_0, max_temp, min_temp, tau):
    print('Performing simulated annealing...')

    # Lists to store working x and y values
    x_vals = list()
    y_vals = list()

    # Add initial x and y
    x_vals.append(x_0)
    y_vals.append(y_0)

    current_temp = max_temp
    time = 0
    current_x, current_y = x_0, y_0
    current_energy = function(current_x, current_y)

    while current_temp > min_temp:

        # Cooling
        time += 1
        current_temp = max_temp * exp(-time / tau)

        # General proposed new point
        proposal_x = current_x + rand.gauss(0, 1)
        proposal_y = current_y + rand.gauss(0, 1)

        # If proposed point is out of bounds, reject it
        if not ((x_bounds[0] < proposal_x < x_bounds[1]) and (y_bounds[0] < proposal_y < y_bounds[1])):
            # Current x and y haven't changed. Append them to the lists
            x_vals.append(current_x)
            y_vals.append(current_y)
            continue

        # Calculate energy of proposed point
        proposal_energy = function(proposal_x, proposal_y)
        delta_energy = proposal_energy - current_energy

        # Check if proposal is accepted
        if rand.random() <= exp(-delta_energy / current_temp):
            # Change current x and y
            current_x = proposal_x
            current_y = proposal_y
            current_energy = proposal_energy

        # Append current values (which may have onw been changed) to lists
        x_vals.append(current_x)
        y_vals.append(current_y)

    return x_vals, y_vals


# ============ PLOTTING ================================================================================================
def gen_plot(x, y, title, save_title):

    plt.figure(figsize=double_fig_size, dpi=dpi)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', figsize=double_fig_size, dpi=dpi)

    ax1.plot(x, ls='', marker='.', c='C0')
    ax2.plot(y, ls='', marker='.', c='C1')

    ax2.set_xlabel(r'$t$ (steps)')

    ax1.set_ylabel(r'$x$')
    ax2.set_ylabel(r'$y$')
    ax1.grid()
    ax2.grid()

    fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_title)
    plt.close()


# ============ QUESTION SETUP ==========================================================================================
# Question 1b
def function_one(x, y):
    return (x ** 2) - cos(4 * pi * x) + ((y - 1) ** 2)


x_0_one = 2
y_0_one = 2
x_bounds_one = (-inf, inf)
y_bounds_one = (-inf, inf)
max_temp_one = 10.0
min_temp_one = 1e-3
tau_one = 1e4


# Question 1c
def function_two(x, y):
    return cos(x) + cos(x * sqrt(2)) + cos(x * sqrt(3)) + ((y - 1) ** 2)


x_0_two = 2
y_0_two = 2
x_bounds_two = (0, 50)
y_bounds_two = (-20, 20)
max_temp_two = 10.0
min_temp_two = 1e-3
tau_two = 1e4

# ============ Simulate annealing and print results ====================================================================
print('============ QUESTION 1B ======================================================================================')
x_vals, y_vals = anneal(function_one, x_bounds_one, y_bounds_one, x_0_one, y_0_one, max_temp_one, min_temp_one, tau_one)
gen_plot(x_vals, y_vals, 'Simulated Annealing for Question 1b', 'Lab11_Q01_b.png')
print(f'Final x (to three decimal places): {np.format_float_scientific(x_vals[-1], precision=3)}')
print(f'Final y (to three decimal places): {np.format_float_scientific(y_vals[-1], precision=3)}')

print()

print('============ QUESTION 1C ======================================================================================')
x_vals, y_vals = anneal(function_two, x_bounds_two, y_bounds_two, x_0_two, y_0_two, max_temp_two, min_temp_two, tau_two)
gen_plot(x_vals, y_vals, 'Simulated Annealing for Question 1c', 'Lab11_Q01_c.png')
print(f'Final x (to three decimal places): {np.format_float_scientific(x_vals[-1], precision=3)}')
print(f'Final y (to three decimal places): {np.format_float_scientific(y_vals[-1], precision=3)}')
