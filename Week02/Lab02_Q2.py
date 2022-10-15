"""
PHY 407: Computational Physics, Lab 02, Question 2
Author: John Wood
Date: September 21, 2022

Script for the pseudocode of questions 2b through e. This script analyses values of a given definite integral defined by
the function f and the bounds [lower_bound, upper_bound]. An exact value is found (pi, from the lab write-up), and
estimates are found using the trapezoidal rule ad Simpson's rule. Associated errors are computed. In question 2c,
the number of slices for each of these approximations to yield an error of order -9 is found. In questions 2d and 2e,
the estimated errors are found and compared to actual errors.

Outputs:
    - Exact definite integral
    - Estimates according to the trapezoidal and Simpson's rules, and associated errors
    - Number of slices required for each method to yield an error of a certain order

"""

# ============ PSEUDOCODE ==============================================================================================
# NOTE: See pseudocode from Lab02_functions.py

# Define function f according to f(x) = 4 / (1 + x ** 2)
# Set lower and upper bounds

# ------ Question 2b ----------------------------------------------------------------------
# Set number of slices, N, as 4
# Compute exact integral of f
# Estimate integral of f with trapezoidal rule, N slices
# Estimate integral of f with Simpson's rule, N slices
# For each rule:
    # Print estimate
    # Print error and relative error between approximation and true value

# ------ Question 2c ----------------------------------------------------------------------
# Set error order as 10 ** -9 (error should be at or below this order)
# For each rule:
    # Set the number of slices, N, as 2
    # Estimate integral of f with the rule. Time this operation
    # Compute the error of the estimate
    # If the the absolute error is greater than the error threshold:
        # Multiply N by 2 and go to line 50
    # Otherwise:
        # Print the error, N, and time

# ------ Question 2d ----------------------------------------------------------------------
# Run error estimation function with 32 slices
# Computer actual error using trapezoidal rule
# Print errors and their differences

# ------ Question 2e ----------------------------------------------------------------------
# Run error estimation function with 32 slices
# Computer actual error using Simpson's
# Print errors and their differences

# ======================================================================================================================

from time import perf_counter_ns
from numpy import pi
import Lab02_functions as labfunc
from math import inf


# ============ SET CONSTANTS ===========================================================================================
def f(x: float) -> float:
    return 4 / (1 + x ** 2)


slices = 4
estimate_slices = 32
lower_bound = 0
upper_bound = 1
error_order = 10 ** -9

# Store integration methods, their names, and error estimation methods in tuples for access with for loops
rules = (labfunc.integrate_trapezoid, labfunc.integrate_simpson)
rule_names = ('Trapezoidal rule', 'Simpson\'s rule')
error_estimations = (labfunc.estimate_trapezoid_error, labfunc.faulty_estimate_simpson_error)

# ============ EXACT INTEGRATION =======================================================================================
integral_exact = pi

# ============ QUESTION 2B =============================================================================================
print('======================== QUESTION 2B ========================')
print('Exact')
print(f'Value: {integral_exact}\n')

# Print estimates and error for each rule
for rule, name in zip(rules, rule_names):
    integral_estimate = rule(slices, lower_bound, upper_bound, f)

    print(name)
    print(f'Estimate: {integral_estimate}')
    print(f'Error: {integral_estimate - integral_exact}')
    print(f'Relative error: {labfunc.relative_error(integral_estimate, integral_exact)}\n')

# ============ QUESTION 2C =============================================================================================
print('======================== QUESTION 2C ========================')
for rule, name in zip(rules, rule_names):
    test_slices = 2
    error = inf # To ensure while loop is entered
    integral_estimate = 0
    time_taken = 0

    # Keep increasing slices then updating error, time, and estimate until threshold met
    # Increase slices at start so checking condition is after estimate update
    while abs(error) > error_order * 10:
        test_slices *= 2

        start_time = perf_counter_ns()
        integral_estimate = rule(test_slices, lower_bound, upper_bound, f)
        end_time = perf_counter_ns()

        error = integral_estimate - integral_exact
        time_taken = end_time - start_time

    print(name)
    print(f'Estimate: {integral_estimate}')
    print(f'Error: {error}')
    print(f'Slices: {test_slices}')
    print(f'Time (ns): {time_taken}\n')

# ============ QUESTION 2D and 2E ======================================================================================
for question, rule, rule_name, estimation in zip(('2D', '2E'), rules, rule_names, error_estimations):
    print(f'======================== QUESTION {question} ========================')
    print(f'{rule_name} for {estimate_slices} slices')

    estimate_error = estimation(estimate_slices, lower_bound, upper_bound, f)
    # Error according to textbook: true_value = estimate + error
    true_error = integral_exact - rule(estimate_slices, lower_bound, upper_bound, f)

    print(f'Error estimate: {estimate_error}')
    print(f'Actual error: {true_error}')
    print(f'Difference in errors: {estimate_error - true_error}')
    print(f'Relative difference in errors: {labfunc.relative_error(estimate_error, true_error)}\n')
