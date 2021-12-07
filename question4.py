import numpy as np
import scipy as sp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import math
import pdb

plt.style.use('seaborn-poster')

x, y = zip(*sorted(zip(np.random.random_sample(size = 6), np.random.random_sample(size = 6))))
f = CubicSpline(x, y, bc_type='natural')
x_test = np.linspace(0, 1, 100)

plt.figure(figsize = (10,8))
plt.plot(x_test, f(x_test), 'g')
plt.plot(x, y, 'ro')
plt.title('Cubic Spline Interpolation')
plt.show()