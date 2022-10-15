"""
PHY 407: Computational Physics, Lab 03, Question 2
Author: John Wood
Date: September 30, 2022

Code for question 2. In question 2a, we compare an estimated relativistic period with the classical limit. This is
done with Gaussian quadrature for N=8 and 16. Errors are estimated. Integrands and weighted integrands are plotted for
question 2b. In question 2d, the fractional error for the period is estimated with N=200.
Finally, in question 2e, x_c is stored as the amplitude for which a classical harmonic oscillator will reach v=c at
x=0. Periods for 1m <= x_0 <= 10 * x_c are then plotted with the integration method (N=200), classical limit,
and relativistic limit.

Outputs:
    - q2b.png: integrands and weighted integrands for N=8 and N=16
    -q2e.png: periods under integration, classical, and relativistic limit

"""

# ============ PSEUDOCODE ==============================================================================================
#
# ------ Generally useful functions ------------------------------------------------------------------------------------
# Set constants (speed of light c, mass m, spring constant k)
# Import Gaussian quadrature functions from gaussxw.py
#
# Define function to compute integrand (g_k)
#     Set initial position x_0 and position x
#     Return integrand according to equations 6 and 7 from lab handbook
#
# Define function to sum integrands according to points and weights:
#     Set points and weights
#     Set total to 0
#     For each corresponding point x and weight w:
#         Add w * integrand(x) to total
#     Return total
#
# Define function to scale points and weights for Gaussian quadrature:
#     Set points and weights
#     Scale points and weights according to equations from pg. 1 of lab handbook
#     Return scaled points and weights
#
# ------ Question 2A/B -------------------------------------------------------------------------------------------------
# Set initial position x_0 as 0.01
# Compute classical period value
#
# Initialize figure for plots
#
# For N = 8 and N = 16:
#     Get points and weights using given subroutine with bounds 0 to x_0
#     Sum integrands with these points and weights to estimate period
#
#     Compute and display error and relative error of estimated period wrt classical period
#     Do same period estimate with 2N slices, considering the resulting estimate as the true value
#     Compute and display the relative error wrt this 'true' value
#
#     Plot integrands vs. points
#     Plot weighted integrands vs. points
#
# Add titles and legends to plot
# Save plot
#
# ------ QUESTION 2D ---------------------------------------------------------------------------------------------------
# Compute points and weights for N=200 and N=400
# Estimate period using both sets of points and weights
#
# Take period estimated using N=400 as actual value
# Compute and display relative error
#
# ------ QUESTION 2E ---------------------------------------------------------------------------------------------------
# Set quantity of initial times as Q
# Compute and display x_c according to equation in lab write-up
# Set first initial position x_0_0 as 1
# Set last initial position x_0_f as 10 * x_c
# Set x_0 as range from x_0_0 to x_0_f (non-inclusive) with Q values
# Initialize estimate, classical, and relativistic period arrays, each with length Q
#
# For each integer 0 <= i <= Q:
#     Scale points and weights for bounds 0 to x_0_i and N=200
#     Using these, compute and store estimated period
#     Compute and store classical limit period
#     Compute and store relativistic limit period
#
# Plot estimate, classical and relativistic periods vs. x_0
# Add titles and legend to plot
# Save plot
#
# ======================================================================================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc
import gaussxw as gxw

# ============ CONSTANTS ===============================================================================================
fig_size = (12, 8)
dpi = 200
plt.rcParams.update({'font.size': 12})
quantity = 2_000

m = 1 # kg
k = 12 # N / kg
c = spc.c # m / s^2
pi = spc.pi


def velocity(x, x_0):
    special_term = 0.5 * k * (x_0 ** 2 - x ** 2)
    numerator = k * (x_0 ** 2 - x ** 2) * (2 * m * c ** 2 + special_term)
    denominator = 2 * (m * c ** 2 + special_term) ** 2

    return c * np.sqrt(numerator / denominator)


def period_integrand(x, x_0):
    return 4 * np.power(velocity(x, x_0), -1)


def integrate_with_weights(points, weights, x_0):
    total = float()
    for point, weight in zip(points, weights):
        total += weight * period_integrand(point, x_0)

    return total


def relative_error(experimental_value, true_value):
    """ Compute the relative error.

    Args:
        experimental_value: The experimental value.
        true_value: The true value.

    Returns:
        The relative error.

    Raises:
        ZeroDivisionError: If the true value is equal to zero.

    """
    if true_value == 0:
        raise ZeroDivisionError('True value is equal to zero.')
    return (experimental_value - true_value) / true_value


def scale_points_and_weights(points, weights, lower_bound, upper_bound):
    points_scaled = 0.5 * (upper_bound - lower_bound) * points + 0.5 * (upper_bound + lower_bound)
    weights_scaled = 0.5 * (upper_bound - lower_bound) * weights
    return points_scaled, weights_scaled


# ============ QUESTION 2A/B ===========================================================================================
print('============ QUESTION 2A/B ================================================')
# Constants
x_0 = 10 ** -2
period_classical = 2 * pi * np.sqrt(m / k)

# Figure initialisation
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, dpi=dpi, figsize=fig_size, tight_layout=True)

for N in (16, 8):
    # Compute integrands and weighted integrands. Sum to estimate period
    points, weights = gxw.gaussxwab(N, 0, x_0)
    integrands = period_integrand(points, x_0)
    weighted_integrands = weights * integrands
    period_estimate = np.sum(weighted_integrands)

    # Relative error
    relative_error_true = relative_error(period_estimate, period_classical)

    # Double the slices to estimate error
    period_estimate_double = integrate_with_weights(*gxw.gaussxwab(2 * N, 0, x_0), x_0)
    relative_error_estimate = relative_error(period_estimate, period_estimate_double)

    # Print results
    print(f'------------ N = {N} ------------------------------------------------')
    print(f'Estimated period: {period_estimate} s')
    print(f'Classical period: {period_classical} s\n')
    print(f'Error (wrt classical): {period_estimate - period_classical} s')
    print(f'Relative error (wrt classical): {relative_error_true} s\n')
    print(f'Relative error estimate: {relative_error_estimate} s\n\n')

    # Plot results
    ax0.bar(points, integrands, width=0.0005, label=f'N={N}')
    ax1.bar(points, weighted_integrands, width=0.0005, label=f'N={N}')

# Finalize axes and save figure
ax0.set_xlabel('x (m)')
ax0.set_ylabel('Integrand (s)')
ax0.legend()

ax1.set_xlabel('x (m)')
ax1.set_ylabel('Weighted integrand (s)')

plt.savefig('q2b.png')

# ============ QUESTION 2D =============================================================================================
print('============ QUESTION 2D ==================================================')
# Get scaled points and weights for N=200, 400. Store unscaled values for N=200 for later use
points_200, weights_200 = gxw.gaussxw(200) # to be used again
points_400_scaled, weights_400_scaled = gxw.gaussxwab(400, 0, x_0) #x_0 still defined from question 2A/B
points_200_scaled, weights_200_scaled = scale_points_and_weights(points_200, weights_200, 0, x_0)

# Estimate period with N=200, 400. Estimate relative error
period_estimate = integrate_with_weights(points_200_scaled, weights_200_scaled, x_0)
period_estimate_double = integrate_with_weights(points_400_scaled, weights_400_scaled, x_0)
relative_error_estimate = relative_error(period_estimate, period_estimate_double)

# Print results
print(f'Relative error estimate: {relative_error_estimate}\n\n')

# ============ QUESTION 2E =============================================================================================
print('============ QUESTION 2E ==================================================')
x_c = c * np.sqrt(1 / 12)# meters
x_0_0 = 1 # first x_0 in meters

print(f'x_c: {x_c} m')

# Initialize arrays for x_0 values and arrays for period estimates
x_0 = np.linspace(x_0_0, 10 * x_c, quantity)
period_estimate = np.zeros(quantity)
period_relativistic = np.zeros(quantity)

# Populate period arrays
print('Populating arrays...')
for i in range(quantity):
    if i % 500 == 0:
        print(f'Progress update: array population at value {i} of {quantity}')

    # Period estimate
    period_estimate[i] = integrate_with_weights(*scale_points_and_weights(points_200, weights_200, 0, x_0[i]), x_0[i])
    # Relativistic limit
    period_relativistic[i] = (4 * x_0[i]) / c

# Plot
plt.figure(2, dpi=dpi, figsize=fig_size)
plt.plot(x_0, period_estimate, label='Estimate')
plt.plot(x_0, np.full(quantity, period_classical), label='Classical limit')
plt.plot(x_0, period_relativistic, label='Relativistic limit')

plt.title('')
plt.xlabel('x_0 (m)')
plt.ylabel('T (s)')
plt.legend()

plt.savefig('q2e.png')
