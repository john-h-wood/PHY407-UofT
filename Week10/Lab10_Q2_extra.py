"""
PHY 407: Computational Physics, Lab 10, Question 02
Author: John Wood
Date: November 29, 2022

Extra script to plot the volume of the n-dimensional unit hypersphere against n.

"""

import numpy as np
from math import factorial
from scipy.special import gamma
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})
dimensions = range(1, 101)
radius = 1

volumes = list()

for d in dimensions:
    true_value = (radius ** d) * (np.pi ** (d / 2))
    if d % 2 == 0:
        true_value /= factorial(d // 2)
    else:
        true_value /= gamma((d / 2) + 1)
    volumes.append(true_value)

plt.figure(dpi=300)
plt.plot(dimensions, volumes)
plt.grid()

plt.xlabel(r'$n$')
plt.ylabel(r'$V_n$')
plt.title(r'Volume of the $n$-dimensional unit hypersphere')

plt.tight_layout()
plt.savefig('Lab10_Q02_volumes.png')
plt.close()
