# Set variables for the gravitational constant and the mass of the Sun
# Create empty arrays for the time, the components of Earth’s position, the components of Earth’s velocity, the components of Jupiter’s position and the components of Jupiter’s velocity.
# Set the initial conditions for Earth’s position and velocity and Jupiter’s position and velocity.
#
# Calculate Jupiter’s orbit:
#   For each time step but the last:
#       For each component:
#           Calculate velocity at next time step according to equation 3
#           Calculate position at new time step according to equation 3
#           Add new velocity and position to their respective arrays
#
# Calculate Earth’s orbit:
#   For each time step but the last:
#       For each component:
#           Calculate velocity at next time step according to equation 3, but this time include Jupiter’s gravitational force
#           Calculate position at new time step according to equation 3
#           Add new velocity and position to their respective arrays
#
# Plot x vs y components of Earth’s position.
# Plot x vs y components of Jupiter’s position.
