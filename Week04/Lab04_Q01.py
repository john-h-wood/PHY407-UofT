"""
PHY 407: Computational Physics, Lab 04, Question 01
Author: John Wood
Date: September 30, 2022

TODO

Outputs:
    - TODO

"""

import numpy as np
import SolveLinear as sl
import matplotlib.pyplot as plt
from time import perf_counter_ns

# ============ PSEUDOCODE ==============================================================================================
# ------------ Question 1A ---------------------------------------------------------------------------------------------
# Import required modules, including that for the pivot and partial pivot methods
# Define the matrix A and vector v according to equation 6.12
#
# Using the partial pivot method, solve the equation Ax=v for x
# Print the solution and expected solution

# ------------ Question 1B ---------------------------------------------------------------------------------------------
# Define a minimum and maximum N
# Define the list of solution methods
#
# For each integer N such that min_N <= N <= max_N:
#   Define an NxN matrix A with random entries
#   Define an N-dimensional vector v with random entries
#
#   For each solution method:
#       Solve the equation Ax=v for x. Time this process
#       Store the time taken
#
#       Compute the dot product of A and x
#       Compute and store the mean of the absolute value of the differences between v and A dot x
#
# Plot the time taken vs. N for each method
# Plot the error vs. N for each method

# ======================================================================================================================

# ============ CONSTANTS ===============================================================================================
fig_size = (12, 8)
dpi = 200
plt.rcParams.update({'font.size': 15})

# ============ QUESTION 1A =============================================================================================
print('============ QUESTION 1A ==================================================')
# Test values
A = np.array([[2, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]], float)
v = np.array([-4, 3, 9, 7], float)

# Print expected values and results
print(f'x from PartialPivot: {sl.PartialPivot(A, v)}')
print('x expected: [2, -1, -2, 1]')

# ============ QUESTION 1B =============================================================================================
print('\n============ QUESTION 1B ==================================================')
# Constants
min_N = 5
max_N = 300
N_range = range(min_N, max_N + 1)
solving_methods = (sl.GaussElim, sl.PartialPivot, np.linalg.solve)
solving_methods_titles = ('Gaussian elimination', 'Partial pivoting', 'LU decomposition')

print(f'Because max N is set to {max_N}, this part of the code takes a few moments to run.')
print('Progress updates are provided.\n')

# Array for timings and errors (dimensions are [time/error], [method], [N])
results = np.empty((2, len(solving_methods), max_N - min_N + 1))

# Generate results
for N in N_range:
    if N % 25 == 0:
        print(f'Progress update: N at {N} of {max_N}')

    # Random data
    A = np.random.rand(N, N)
    v = np.random.rand(N)

    # Solve, time and check error for each method
    for i, method in enumerate(solving_methods):
        start_time = perf_counter_ns()
        x = method(A, v)
        end_time = perf_counter_ns()
        results[0, i, N - min_N] = end_time - start_time

        v_sol = np.dot(A, x)
        results[1, i, N - min_N] = np.mean(np.abs(v - v_sol))

# Plotting times
plt.figure(0, dpi=dpi, figsize=fig_size)
for i, title in enumerate(solving_methods_titles):
    plt.plot(N_range, results[0, i, :], label=title)

plt.yscale('log')
plt.xlabel('N')
plt.ylabel('Time (ns)')
plt.legend()
plt.savefig('LAB04_Q01_B_1.png')

# Plotting errors
plt.figure(1, dpi=dpi, figsize=fig_size)
for i, title in enumerate(solving_methods_titles):
    plt.plot(N_range, results[1, i, :], label=title)

plt.yscale('log')
plt.xlabel('N')
plt.ylabel('Mean error')
plt.legend()
plt.savefig('LAB04_Q01_B_2.png')
