import numpy as np
from scipy.interpolate import Rbf

# Generate some random multidimensional data
np.random.seed(0)
x = np.random.rand(100, 3)  # 100 points in 3 dimensions
y = np.sin(2 * np.pi * x[:, 0]) + np.cos(2 * np.pi * x[:, 1]) + np.sin(2 * np.pi * x[:, 2])

# Define the grid on which to interpolate
grid_x, grid_y, grid_z = np.mgrid[0:1:100j, 0:1:100j, 0:1:100j]

# Perform RBF interpolation
rbf = Rbf(x[:, 0], x[:, 1], x[:, 2], y, function='gaussian')
interpolated_values = rbf(grid_x, grid_y, grid_z)

# Example of evaluating the interpolated values at a specific point
point = (0.5, 0.5, 0.5)
interpolated_value_at_point = rbf(*point)

print("Interpolated value at point {}: {}".format(point, interpolated_value_at_point))