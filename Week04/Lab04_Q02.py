"""
PHY 407: Computational Physics, Lab 04, Question 02
Author: John Wood
Date: September 30, 2022

Uses linear algebra to solve for the energies and wavefunctions of an electron is a asymmetric potential well.
Equations from the lab handout and Computational Physics are implemented to calculate a Hamiltonian matrix.
Eigenvectors and values are found and used to display energies and probability distributions.

Outputs:
    - Printed: Energies and errors
    - L

"""

import numpy as np
import lab_func as labfunc
import scipy.constants as spc
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh, eigh

# ============ PSEUDOCODE ==============================================================================================
# ------------ Question 2B ---------------------------------------------------------------------------------------------
# Define the well width, L
# Define the constant a = 10 eV
# Define h_bar and the mass of the electron
#
# Define function to compute elements of H:
#   Set m and n
#   Determine whether m=n, they are both odd/even, or one is even and the other is odd
#   With this, return the value of H_{m, n} according to Equation 5 from the lab handout
#
# ------------ Question 2C/D -------------------------------------------------------------------------------------------
# Define function to compute H matrix:
#   Set row and column number
#   Define an empty matrix of the correct dimensions
#   For each row m:
#       For each column n:
#           Set the matrix element (m, n) to be H(m, n) with the function from Question 2B
#   Return the matrix
#
#   Create a 10x10 H matrix
#   Compute, store, and print the first 10 eigenvalues of this matrix
#   Create a 100x100 H matrix
#
#   Compute, store, and print the first 10 eigenvalues of this matrix. Also compute and store the eigenvectors,
#       for later use
#
#   For each eigenvalue from the smaller matrix, calculate its relative error with respect to the corresponding
#       eigenvalue from the 100x100 matrix
#
#   Print these relative errors
#
# ------------ Question 2E ---------------------------------------------------------------------------------------------
#   Define a function to compute a probability density:
#       Set the eigenvector with Fourier coefficients
#       Set the x values for which to calculate the function
#       Define empty results array with the length of the x array
#
#       For each integer n such that 1 <= n <= [dimension of the eigenvector]:
#           Find n-th component of the eigenvector
#           Add Fourier term to results array according to Equation at top of page 249 in Computational Physics
#       Return results array squared
#
#   Compute and store points and weights for Gaussian quadrature from 0 to L with 250 points
#   For each integer i such that 0 <= i <= 2:
#       Compute the probability density with the i-th eigenvector from Question 2D
#       Integrate this probability density using Gaussian quadrature
#       Plot the probability density divided by its integral
#   Display plot
#
# ======================================================================================================================

# ============ CONSTANTS ===============================================================================================
# Numpy printing options
precision = 4
np.set_printoptions(precision=precision)

# Plotting options
fig_size = (12, 8)
dpi = 300
plt.rcParams.update({'font.size': 15})

# All in SI units
# L = 5 * spc.angstrom # m
# a = 10 * spc.electron_volt # J
# M = spc.electron_mass # kg
# h_bar = spc.hbar # Js s
# pi = spc.pi

L = 5
a = 10
pi = spc.pi
h_bar = spc.physical_constants['reduced Planck constant in eV s'][0]
mass_unit = (spc.electron_volt ** -1) * (spc.angstrom ** 2)
M = spc.electron_mass * mass_unit


# ============ QUESTION 2B =============================================================================================
def h_matrix_element(m: int, n: int) -> float:
    """
    Computes a single element of the H matrix according to Equation 5 of the lab document.

    Args:
        m: The row of the element.
        n: The column of the element.

    Returns:
        The element.

    """
    diff = abs(m - n) # use absolute difference between m and n to determine if they are equal, both odd/even, etc.

    if diff == 0:
        # m and n are equal
        return (0.5 * a) + (((pi ** 2) * (h_bar ** 2) * (m ** 2)) / (2 * M * (L ** 2)))
    elif diff % 2 == 1:
        # exactly one of m and n are odd
        return -1 * ((8 * a * m * n) / ((pi ** 2) * (((m ** 2) - (n ** 2)) ** 2)))
    else:
        # m and n are both even or both odd
        return 0.


# ============ QUESTION 2C/D ===========================================================================================
print('============ QUESTION 2C/D ====================================================================================')


def h_matrix(max_m: int, max_n: int) -> np.ndarray:
    """
    Computes an H matrix for a given size.

    Args:
        max_m: The number of rows.
        max_n: The number of columns.

    Returns:
        The matrix.

    """
    h = np.empty((max_m, max_n))

    for m in range(1, max_m + 1):
        for n in range(1, max_n + 1):
            # ensure that m and n start from 1, but that indices start at 0
            h[m - 1, n - 1] = h_matrix_element(m, n)

    return h


# Print results for 10x10 matrix
h_10x10 = h_matrix(10, 10)
# First ten energies in eV
first_energies_10x10 = eigvalsh(h_10x10)[:10]

print(f'First ten energies (eV) (with H at 10x10):\n {first_energies_10x10}\n')
print(f'Check: According to Newman, ground-state energy should be about 5.84 eV. It is '
      f'{np.format_float_scientific(first_energies_10x10[0], precision)} eV.')

# ============ QUESTION 2D =============================================================================================
print()
print('============ QUESTION 2D ======================================================================================')
# Print results for 100x100 matrix and errors
h_100x100 = h_matrix(100, 100)
# Store eigenvectors for use in question 2E
energies_100x100, eigenvectors_100x100 = eigh(h_100x100)
# First ten energies in eV
first_energies_100x100 = energies_100x100[:10]

print(f'First ten energies (eV) (with H at 100x100):\n {first_energies_100x100}\n')
print('Relative error of energies from H 10x10 wrt H 100x100:')
print(labfunc.relative_error(first_energies_10x10, first_energies_100x100))


# ============ QUESTION 2E =============================================================================================
def prob_density(eigenvector_in: np.ndarray, x_in: np.ndarray) -> np.ndarray:
    wave_function = np.empty(len(x_in), float)

    for n, fourier_coefficient in enumerate(eigenvector_in):
        wave_function += fourier_coefficient * np.sin(pi * (n + 1) * x_in * (1 / L))

    return wave_function ** 2


# Points and weights for Gaussian integration
points, weights = labfunc.gaussxwab(250, 0, L)
# Points for plotting
x = np.linspace(0, L, 200)

# Compute, integrate, normalize, and plot the probability densities for the first three energies
# Initialize plot
plt.figure(0, figsize=fig_size, dpi=dpi)
for i in range(3):
    eigenvector = eigenvectors_100x100[:, i]
    integral = np.sum(weights * prob_density(eigenvector, points))

    psi_squared_normed = prob_density(eigenvector, x) / integral
    plt.plot(x, psi_squared_normed, label=f'E{i}')

plt.xlabel('Position (Å)')
plt.ylabel('$|\psi|^2$ (Å$^{-1}$)')
plt.legend()
plt.savefig('LAB04_Q02.png')
