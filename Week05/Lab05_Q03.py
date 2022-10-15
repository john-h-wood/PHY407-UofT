"""
PHY 407: Computational Physics, Lab 05, Question 03
Author: John Wood
Date: October 15, 2022

TODO

Outputs:
    - TODO

"""
import numpy as np
import matplotlib.pyplot as plt

# ============ QUESTION 2A =============================================================================================
# Extract data
longitude = np.loadtxt('lon.txt')
time = np.loadtxt('times.txt')
slp = np.loadtxt('SLP.txt')
