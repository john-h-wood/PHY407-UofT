"""
PHY 407: Computational Physics, Lab 10, Question 02
Author: John Wood
Date: November 29, 2022

*** NOTE ***
Numpy recommends using the PCG64 pseudorandom number generation algorithm.
See https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
Here, I use its legacy Mersenne Twister algorithm to stay consistent with Python's built-in random module.

This script approximates the volume of an n-dimensional sphere using the mean value method of Monte Carlo
Integration. The true value is also computed. Results and relative error are printed.

Outputs:
    - Printed: volume estimate in units^dimensions (i.e. units squared, or units cubed, etc.)
    - Printed: the true volume in the same units
    - Printed: the relative error in the same units

"""

# ============ PSEUDOCODE ==============================================================================================
# Import required modules: numpy (including random number generation), SciPy's gamma function, and the factorial
#     function
# Set number of dimensions, radius and number of sample points
# Calculate the volume of the hypercube over which we are integrating
#
# Define function for relative error:
#     Set experimental value (ev) and true value (tv)
#     Return (ev - tv) / tv
#
# Set number of points inside sphere to zero
# For each integer i in [0, number of sample points]:
#     Generate [dimensions] random numbers between -radius and radius
#     Calculate sum of squares of the random numbers
#     If this sum is less than or equal to radius squares:
#         Add one to number of points inside sphere
# Calculate and print volume estimate using Equation 10.33 from Newman
# Calculate and print true value using Equation 2 from lab handout
# Calculate and print relative error using function from above
#
# Print estimate, true value, and relative error, all rounded to 3 decimals
#
# ======================================================================================================================

import numpy as np
from math import factorial
from scipy.special import gamma

# ============ CONSTANTS ===============================================================================================
dimensions = 10
sample_points = 1_000_000
radius = 1
container_volume = (2 * radius) ** dimensions


def relative_error(experimental_value: float, actual_value: float) -> float:
    """ Compute the relative error.

    Args:
        experimental_value: The experimental value.
        actual_value: The actual value.

    Returns:
        The relative error.

    Raises:
        ZeroDivisionError: If the actual value is equal to zero.

    """
    if true_value == 0:
        raise ZeroDivisionError('Actual value is equal to zero.')
    return (experimental_value - actual_value) / actual_value


# ============ MONTE-CARLO INTEGRATION =================================================================================
print('============ MONTE CARLO INTEGRATION ==========================================================================')
points_in_sphere = 0

# For each sample point, generate random n-dimensional point and test if it's in the sphere
for i in range(1, sample_points + 1):
    if i % 100_000 == 0:
        print(f'Working on point {i // 1_000}-thousand of {sample_points}')
    random_point = (2 * radius) * np.random.random_sample(dimensions) - radius
    if np.sum(np.square(random_point)) <= radius ** 2:
        points_in_sphere += 1

print('\nDone integration.\n')


volume_estimate = (container_volume / sample_points) * points_in_sphere # see Eq 10.33 on pg. 470 of Newman

# Compute true value, but use factorial instead of gamma is dimensions is even
true_value = (radius ** dimensions) * (np.pi ** (dimensions / 2))
if dimensions % 2 == 0:
    true_value /= factorial(dimensions // 2)
else:
    true_value /= gamma((dimensions / 2) + 1)

# Print results
print('============ RESULTS ==========================================================================================')
print(f'Estimate of value for a {dimensions}-dimensional sphere: {volume_estimate}')
print(f'True value: {true_value}')
rel_error = relative_error(volume_estimate, true_value)
print(f'Relative error: {rel_error}')
print()
print('Results rounded to 3 decimals ---------------------------------------------------------------------------------')
print(f'Estimate value: {np.format_float_scientific(volume_estimate, 3)}')
print(f'True value: {np.format_float_scientific(true_value, 3)}')
print(f'Relative error: {np.format_float_scientific(rel_error, 3)}')
