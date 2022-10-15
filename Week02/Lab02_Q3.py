"""
PHY 407: Computational Physics, Lab 02, Question 3
Author: John Wood
Date: September 21, 2022

Script for the pseudocode of questions 3. This script integrates the blackbody function over all wave numbers using
Simpson's rule. This finds the total work per unit area, W. In question 3b, W is found for an example temperature,
100K. Question 3c estimates the value of the Stefan-Boltzmann constant using T=1K. Errors are computed. The error of
the numerical integration is approximated using the method described in Newman's Computational Physics.
"""

# ============ PSEUDOCODE ==============================================================================================
# NOTE: As required, this pseudocode is only for the code of part b
# NOTE: See pseudocode from Lab02_functions.py

# Define function f according to equation from lab write-up, that is,
#   f(z) = (z / (1 - z)) ** 3) * ((exp(z / (1 - z)) - 1) ** (-1)) * ((1 / (1 - z)) ** 2)
# Set temperature in Kelvin as T
# Define constant C1 according to equation from lab write-up, that is,
#   C1 = (2 * pi * k^4 * T^4) / (h^3 * c^2)
# Set upper and lower bounds
# Set number of slices

# Estimate integral of f from lower to upper bound using Simpson's rule and the given number of slices
# Multiply this estimate by C1
# Estimate the error for given number of slices, bounds, and f
# Print the (multiplied) estimate and error

# ======================================================================================================================

import Lab02_functions as labfunc
import scipy.constants as spc
from math import exp

def f(z: float) -> float:
    return ((z / (1 - z)) ** 3) * ((exp(z / (1 - z)) - 1) ** (-1)) * ((1 / (1 - z)) ** 2)


slices = 1000
lower_bound = 2 * 10 ** -16
upper_bound = 0.998593


# Finding W according to a temperature is written as a function for later use with finding the Stefan-Boltzmann constant
def find_w(temperature: float):
    integral_constant = (2 * spc.pi * (spc.k ** 4) * (temperature ** 4)) / ((spc.h ** 3) * (spc.c ** 2))
    integral_estimate = labfunc.integrate_simpson(slices, lower_bound, upper_bound, f)

    return integral_constant * integral_estimate


# ============ QUESTION 3B =============================================================================================
print('======================== QUESTION 3B ========================')
example_temperature = 100 # K
print(f'Example calculation: For {example_temperature}K, W is {find_w(example_temperature)}')
print(f'Estimated error: {labfunc.estimate_simpson_error(slices, lower_bound, upper_bound, f)}\n')

# ============ QUESTION 3C =============================================================================================
print('======================== QUESTION 3C ========================')
estimated_value = find_w(1)
estimated_error = labfunc.estimate_simpson_error(slices, lower_bound, upper_bound, f)
actual_value = spc.sigma
print('All values in SI units')
print(f'Estimated value: {estimated_value}')
print(f'Estimated error: {estimated_error}')
print(f'Actual value: {actual_value}')
print(f'Error: {estimated_value - actual_value}')
print(f'Relative error: {labfunc.relative_error(estimated_value, actual_value)}')







