"""
PHY 407: Computational Physics, Lab 02, Question 2
Author: John Wood
Date: September 21, 2022

Useful functions for lab 2.

"""
from typing import Callable

# ============ PSEUDOCODE ==============================================================================================
# Define function for trapezoidal rule:
    # Set number of slices (N), bounds (a, b), and mathematical function (f)
    # From bounds and N, compute width of each slice, h
    # Set area as 0.5 f(a) + 0.5 f(b)
    # For each natural number 1 <= k <= N-1:
        # Add f(a + k * h) to area
    # Return h * area

# Define function for Simpson's rule:
    # Set number of slices (N), bounds (a, b), and mathematical function (f)
    # If number of slices is odd, raise error
    # From bounds and n, compute width of each slice, h
    # Set area as f(a) + f(b)
    # For each odd number 1 <= k <= N-1:
        # Add 4 * f(a + k * h) to area
    # For each even number 2 <= k <= N-2:
        # Add 2 * f(a + k * h) to area
    # Return (1/3) * h * area

# Define function for relative error:
    # Set experimental value (ev) and true value (tv)
    # Return (ev - tv) / tv

# Define function for practical estimation of error
    # Set numbers of slices, N2, and N1 = N2 / 2
    # If number of slices, N2, is odd, raise error
    # Set mathematical function f and bounds (a, b)
    # Compute area using trapezoidal rule and N2 (t2)
    # Compute area using trapezoidal rule and N1 (t1)
    # Return (1 / 3) * (t2 - t1)

# ======================================================================================================================


def integrate_trapezoid(slices: int, lower_bound: float, upper_bound: float, function: Callable[[float], float]) -> \
        float:
    """Estimate a definite integral using the trapezoidal rule.

    Uses equation 5.3 from Newman's Computational Physics.

    Args:
        slices: The number of slices.
        lower_bound: The lower bound.
        upper_bound: The upper bound.
        function: The mathematical function, as a Python function. Must have a single float as parameter and return a
                  float.

    Returns:
        The estimate.

    """
    slice_width = (upper_bound - lower_bound) / slices
    area = 0.5 * (function(lower_bound) + function(upper_bound))

    for k in range(1, slices):
        area += function(lower_bound + k * slice_width)

    return slice_width * area


def integrate_simpson(slices: int, lower_bound: float, upper_bound: float, function: Callable[[float], float]) -> float:
    """Estimate a definite integral using Simpson's rule.

    Uses equation 5.9 from Newman's Computational Physics.

    Args:
        slices: The number of slices.
        lower_bound: The lower bound.
        upper_bound: The upper bound.
        function: The mathematical function, as a Python function. Must have a single float as parameter and return a
                  float.

    Returns:
        The estimate

    Raises:
        ValueError: If the number of slices is odd.

    """
    if slices % 2 != 0:
        raise ValueError('The number of slices must be even for Simpson\'s rule')

    slice_width = (upper_bound - lower_bound) / slices
    area = function(lower_bound) + function(upper_bound)

    for k in range(1, slices, 2):
        area += 4 * function(lower_bound + k * slice_width)
    for k in range(2, slices, 2):
        area += 2 * function(lower_bound + k * slice_width)

    return (1/3) * slice_width * area


def estimate_trapezoid_error(slices: int, lower_bound: float, upper_bound: float, function: Callable[[float],
                                                                                                     float]) -> float:
    """Estimate error of definite integral under trapezoidal rule.

    Uses equation 5.28 from Newman's Computational Physics.

    Args:
        slices: The number of slices of the estimation.
        lower_bound: The lower bound.
        upper_bound: The upper bound.
        function: The mathematical function, as a Python function. Must have a single float as parameter and return a
                  float.

    Returns:
        The error estimate.

    Raises:
        ValueError: If the number of slices is odd.

    """
    if slices % 2 != 0:
        raise ValueError('The number of slices must be even to estimate error.')

    area_2 = integrate_trapezoid(slices, lower_bound, upper_bound, function)
    area_1 = integrate_trapezoid(slices // 2, lower_bound, upper_bound, function)

    return (1 / 3) * (area_2 - area_1)


def faulty_estimate_simpson_error(slices: int, lower_bound: float, upper_bound: float, function: Callable[[float],
                                                                                                     float]) -> float:
    """Faulty function to estimate error of definite integral under Simpson's rule.

    To be used only for question 2e to illustrate why this estimation does not work. Modifies equation 5.28 from
    Newman's Computational Physics.

    Args:
        slices: The number of slices of the estimation.
        lower_bound: The lower bound.
        upper_bound: The upper bound.
        function: The mathematical function, as a Python function. Must have a single float as parameter and return a
                  float.

    Returns:
        The error estimate.

    Raises:
        ValueError: If the number of slices is odd.

    """
    if slices % 2 != 0:
        raise ValueError('The number of slices must be even to estimate error.')

    area_2 = integrate_simpson(slices, lower_bound, upper_bound, function)
    area_1 = integrate_simpson(slices // 2, lower_bound, upper_bound, function)

    return (1 / 3) * (area_2 - area_1)


def estimate_simpson_error(slices: int, lower_bound: float, upper_bound: float, function: Callable[[float],
                                                                                                   float]) -> float:
    """Estimate error of definite integral under Simpson's rule.

        Uses equation 5.29 from Newman's Computational Physics.

        Args:
            slices: The number of slices of the estimation.
            lower_bound: The lower bound.
            upper_bound: The upper bound.
            function: The mathematical function, as a Python function. Must have a single float as parameter and
                      return a float.

        Returns:
            The error estimate.

        Raises:
            ValueError: If the number of slices is odd.

        """
    if slices % 2 != 0:
        raise ValueError('The number of slices must be even to estimate error.')

    area_2 = integrate_simpson(slices, lower_bound, upper_bound, function)
    area_1 = integrate_simpson(slices // 2, lower_bound, upper_bound, function)

    return (1 / 15) * (area_2 - area_1)

def relative_error(experimental_value: float, true_value: float) -> float:
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
