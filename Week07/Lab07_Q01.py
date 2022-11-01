"""
PHY 407: Computational Physics, Lab 07, Question 01
Author: John Wood
Date: November 1, 2022

Plots the trajectory of a ball bering orbiting a rod, as calculated by an adaptive and non-adaptive RK4 algorithm (
wrt step size). Each algorithm is times. For the adaptive algorithm, step size is plotted over time, and also with
velocity.

Outputs:
    - LAB07_Q01_a.png: Trajectories according to adaptive and non-adaptive RK4 algorithms
    - Printed: time taken for each algorithm
    - LAB07_Q01_c.png: Step size over time
    - LAB07_Q01_c2.png: Step size and velocity over time

"""

# ============ PSEUDOCODE ==============================================================================================
#
# Set plotting preferences (dpi, figure size, and font size)
# Set problem constants (mass and length of rod, initial and final time, initial conditions, etc.)
#
# Define function to calculate the four seperated equations for the differential equation:
#     Set x and y position and velocity
#     Calculate x and y velocity and x and y force
#     Return these values
#
# Define function for one or two steps in the RK4 algorithm:
#     Define step quantity and size
#     Define starting conditions and equation of motion
#     Compute k1 through k4 for each of the starting conditions
#     From these k values, find the next set of conditions
#
#     If step quantity is 1:
#         Return this next set
#     If step quantity is 2:
#         Pass this next set into the function recursively and return the result
#     Otherwise:
#         Raise error
#
# Define function for adaptive RK4 algorithm:
#     Set target error, integration times, initial conditions, equation of motion and starting step size, h
#     Set empty time list
#     Set empty step size list
#     Set empty results list (entries formatted as (x, v_x, y, v_y))
#     Set time as integration start time
#     Initialize conditions array as initial conditions
#
#     While time < integration end time:
#
#         Using previous function, find conditions after two h steps
#         Using previous function, find conditions after one 2*h step
#         Using these results, calculate the error in x and y position according to Eq. 8.54 of Computational Physics
#         Calculate rho according to equation 4 from lab handout
#
#         If rho >= 1:
#             If rho > 1:
#                 Calculate increased step size according to Eq. 8.52 from Computational Physics
#                 If increased step size is not greater than 2 * [current step size]:
#                     Set increased step size as step size
#                 Otherwise:
#                     Set 2 * [current step size] as step size
#
#             Add 2 * [current step size] to time
#             Set conditions as those from two h steps
#             Append conditions to results list
#             Append time to time list
#             Append current step size to step size list
#
#         Otherwise (if rho < 1):
#             Calculate smaller step size according to Eq. 8.52 from Computational Physics
#             Set current step size as this smaller step size
#             Calculate step of RK4 method with previously defined function
#
#             Add step size to time
#             Set conditions as those RK4 step
#             Append conditions to results list
#             Append time to time list
#             Append current step size to step size list
#
# Define function for non-adaptive RK4 algorithm
#     Set integration times, step size, and equation of motion
#     Set empty results list
#     Set time range according to integration times and step size
#     Using previously defined RK4 step function, populate results array with recursively passed conditions arrays
#
# ------------ Question 1A ---------------------------------------------------------------------------------------------
# Using problem constants from above, use non-adaptive RK4 algorithm to find trajectory of ball bering. Time this
# Do the same for the adaptive RK4 algorithm'
#
# Plot these trajectories on the same x vs. y plot
#
# ------------ Question 1B ---------------------------------------------------------------------------------------------
# Print the times from Question 1A
#
# ------------ Question 1C ---------------------------------------------------------------------------------------------
# Plot step size vs. time for adaptive RK4 from results of Question 1A.
# Calculate velocity over time from results of Question 1A.
# Plot step size and velocity vs. time for adaptive RK4 from results of Question 1A.
#
# =======================================================================================================================

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter

# ============ CONSTANTS ===============================================================================================
# Plotting preferences
fig_size = (10.5, 8)
dpi = 300
plt.rcParams.update({'font.size': 15})

# Problem constants
G = 1
rod_mass = 10
rod_length = 2
t_0 = 0
t_final = 10
initial_conditions = np.array([1., 0., 0., 1.], float)
delta = 10e-6

use_stepping_condition = True


def seperated_equations(conditions):
    """
    See pseudo-code (lines 23-26). Adapted from Newman_8-8.py by Nico Grisouard.
    """
    v_x = conditions[1]
    v_y = conditions[3]
    x, y = conditions[0], conditions[2]
    r2 = x ** 2 + y ** 2

    f_x, f_y = - rod_mass * np.array([x, y], float) / (r2 * np.sqrt(r2 + .25 * rod_length ** 2))

    return np.array([v_x, f_x, v_y, f_y], float)


# ============ ADAPTIVE RK4 ============================================================================================
# ------------ NOTE ------------
# These functions rely on the functions to solve having no dependence on time. Conditions are given as arrays with
# the format (x, v_x, y, v_y) for each time step.

def rk4_step(step_quantity, step_size, starting_conditions, function):
    """
    See pseudo-code (lines 28-32). Adapted from Newman_8-8.py by Nico Grisouard.
    """
    k1 = step_size * function(starting_conditions)  # all the k's are vectors
    k2 = step_size * function(starting_conditions + 0.5 * k1)  # note: no explicit dependence on time of the RHSs
    k3 = step_size * function(starting_conditions + 0.5 * k2)
    k4 = step_size * function(starting_conditions + k3)

    results = starting_conditions + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    if step_quantity == 1:
        return results
    elif step_quantity == 2:
        return rk4_step(1, step_size, results, function)
    else:
        raise ValueError('Only step quantity 1 and 2 are supported')


def adaptive_rk4(t_0, t_final, step_size, initial_conditions, target_error, function):
    """
    See pseudo-code (lines 41-79).
    """
    conditions = initial_conditions
    t = t_0
    time = list()
    results = list()
    step_sizes = list()

    while t < t_final:
        # Do two steps of step_size and one of 2 * step_size, find the error in each, then compute rho
        two_steps = rk4_step(2, step_size, conditions, function)
        single_step = rk4_step(1, 2 * step_size, conditions, function)
        error_x = (1 / 30) * (two_steps[0] - single_step[0])
        error_y = (1 / 30) * (two_steps[2] - single_step[2])
        rho = (step_size * target_error) / sqrt(error_x ** 2 + error_y ** 2)

        if rho >= 1:  # Desired accuracy achieved. Append results.

            if rho > 1:  # Step size can be increased
                # Compute new step size. If it meets the condition to avoid divergence, set it as the step size
                new_step_size = step_size * (rho ** 0.25)
                if use_stepping_condition:
                    if new_step_size <= 2 * step_size:
                        step_size = new_step_size
                    else:
                        step_size = 2 * step_size
                else:
                    step_size = new_step_size

            t += 2 * step_size
            conditions = two_steps

            time.append(t)
            results.append(two_steps)
            step_sizes.append(step_size)

            continue

        elif rho < 1:  # Desired accuracy not achieved. Do step with required step_size
            new_step_size = step_size * (rho ** 0.25)
            step_size = new_step_size
            new_conditions = rk4_step(1, step_size, conditions, function)

            t += step_size
            conditions = new_conditions

            time.append(t)
            results.append(new_conditions)
            step_sizes.append(step_size)

            continue

    return np.array(time), np.array(results), np.array(step_sizes)


# ============ NON-ADAPTIVE RK4 ========================================================================================
# ------------ NOTE ------------
# These functions again rely on the functions to solve having no dependence on time. Conditions are given as arrays with
# the format (x, v_x, y, v_y) for each time step.
def non_adaptive_rk4(t_0, t_final, step_size, initial_conditions, function):
    """
    See pseudo-code (lines 81-85). Adapted from Newman_8-8.py by Nico Grisouard.
    """
    time = np.arange(t_0, t_final, step_size)
    conditions = initial_conditions
    results = list()

    for t in time:
        results.append(conditions)
        conditions = rk4_step(1, step_size, conditions, function)

    return time, np.array(results)


# ============ QUESTION 1 ==============================================================================================
non_adaptive_start = perf_counter()
non_adaptive_time, non_adaptive_conditions = non_adaptive_rk4(t_0, t_final, 0.001, initial_conditions,
                                                              seperated_equations)
non_adaptive_end = perf_counter()

adaptive_start = perf_counter()
adaptive_time, adaptive_conditions, step_sizes = adaptive_rk4(t_0, t_final, 0.01, initial_conditions, delta,
                                                              seperated_equations)
adaptive_end = perf_counter()


print(len(adaptive_time))
print(len(non_adaptive_time))

# ============ QUESTION 1A =============================================================================================
print('============ QUESTION 1A ======================================================================================')
print('Plotting...')
plt.figure(figsize=fig_size, dpi=dpi)

plt.plot(non_adaptive_conditions[:, 0], non_adaptive_conditions[:, 2], '-.', label='Non-adaptive RK4')
plt.plot(adaptive_conditions[:, 0], adaptive_conditions[:, 2], '--', label=f'Adaptive RK4 ($\delta=$ {delta})')

plt.xlabel('$x$')
plt.ylabel('$y$')
# Note: title is the same as from Newman_8-8.py by Nico Grisouard.
plt.title('Trajectory of a ball bering around a space rod')
plt.axis('equal')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('LAB07_Q01_a.png')

# ============ QUESTION 1B =============================================================================================
print()
print('============ QUESTION 1B ======================================================================================')
print(f'Time for non-adaptive RK4: {np.format_float_scientific(non_adaptive_end - non_adaptive_start, 3)} (s)')
print(f'Time for adaptive RK4: {np.format_float_scientific(adaptive_end - adaptive_start, 3)} (s)')

# ============ QUESTION 1C =============================================================================================
print()
print('============ QUESTION 1C ======================================================================================')
print('Plotting...')
# Step size as a function of time
plt.figure(figsize=fig_size, dpi=dpi)

plt.plot(adaptive_time, step_sizes)
plt.xlabel('Time')
plt.ylabel('Step size')
plt.title('Step size as a function of time under adaptive RK4')
plt.grid()
plt.tight_layout()
plt.savefig('LAB07_Q01_c.png')

# Step size and velocity as a function of time
plt.figure(figsize=fig_size, dpi=dpi)

velocity = np.sqrt(adaptive_conditions[:, 1] ** 2 + adaptive_conditions[:, 3] ** 2)
plt.plot(adaptive_time, step_sizes * 100, label='Step size')
plt.plot(adaptive_time, velocity, label='Velocity')
plt.xlabel('Time (s)')
plt.ylabel('100 $\cdot$ Step size / Velocity')
plt.title('Step size and velocity as a function of time under adaptive RK4')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('LAB07_Q01_c_2.png')
