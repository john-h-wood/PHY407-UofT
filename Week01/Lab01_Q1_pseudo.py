# Set gravitational constant, mass of sun, time step, end time and initial conditions
# Create time array from initial time to end time with interval time step
# Create empty position and velocity arrays for each component (x, y)
# Set initial positions and velocities
#
# For each time step but the last:
#     For each component:
#         Calculate velocity at next time step according to equation 3
#         Calculate position at new time step according to equation 3
#         Add new velocity and position to their respective arrays
#
# Plot x vs. t with y vs t
# Plot x vs. y
