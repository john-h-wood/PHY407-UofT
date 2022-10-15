# ====== GENERALLY USEFUL FUNCTIONS ======================================================
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

# ====== QUESTION 2 =======================================================================
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
