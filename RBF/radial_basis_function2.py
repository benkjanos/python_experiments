import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import Rbf

def generate_data(num_points):
    np.random.seed(0)
    x = np.random.rand(num_points, 5)  # 100 points in 5 dimensions
    y = np.sin(2 * np.pi * x[:, 0]) + np.cos(2 * np.pi * x[:, 1]) + np.sin(2 * np.pi * x[:, 2]) + np.cos(2 * np.pi * x[:, 3]) + np.sin(2 * np.pi * x[:, 4])
    return x, y

def adaptive_refinement(x, y, threshold, max_iterations):
    for _ in range(max_iterations):
        rbf = Rbf(*[x[:, i] for i in range(x.shape[1])], y, function='gaussian')
        interpolated_values = rbf(*[x[:, i] for i in range(x.shape[1])])
        errors = np.abs(interpolated_values - y)
        max_error_index = np.argmax(errors)
        max_error = errors[max_error_index]

        if max_error < threshold:
            break

        new_x = np.vstack([x, x[max_error_index]])
        new_y = np.append(y, y[max_error_index])
        x, y = new_x, new_y

    return x, y

# Generate initial data
x, y = generate_data(100)

# Perform adaptive refinement
refined_x, refined_y = adaptive_refinement(x, y, threshold=0.1, max_iterations=10)

# Define the grid on which to interpolate
grid_shape = (8, 8, 8, 8, 8)
grid_ranges = [(0, 1)] * 5
grid_x = np.linspace(*grid_ranges[0], grid_shape[0])
grid_y = np.linspace(*grid_ranges[1], grid_shape[1])
grid_z = np.linspace(*grid_ranges[2], grid_shape[2])
grid_w = np.linspace(*grid_ranges[3], grid_shape[3])
grid_v = np.linspace(*grid_ranges[4], grid_shape[4])
grid_x, grid_y, grid_z, grid_w, grid_v = np.meshgrid(grid_x, grid_y, grid_z, grid_w, grid_v, indexing='ij')

# Perform RBF interpolation
rbf = Rbf(refined_x[:, 0], refined_x[:, 1], refined_x[:, 2], refined_x[:, 3], refined_x[:, 4], refined_y, function='gaussian')
interpolated_values = rbf(grid_x, grid_y, grid_z, grid_w, grid_v)

# Example of evaluating the interpolated values at a specific point
point = (0.5, 0.5, 0.5, 0.5, 0.5)
interpolated_value_at_point = rbf(*point)

print("Interpolated value at point {}: {}".format(point, interpolated_value_at_point))

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(refined_x[:, 0], refined_x[:, 1], refined_x[:, 2], c=refined_y, cmap='viridis', label='Data Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Data Points')
plt.show()

plt.figure()
plt.imshow(interpolated_values[:, :, grid_shape[2] // 2, grid_shape[3] // 2, grid_shape[4] // 2], extent=(grid_ranges[0][0], grid_ranges[0][1], grid_ranges[1][0], grid_ranges[1][1]), origin='lower')
plt.colorbar(label='Interpolated Value')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Interpolated Values (Z = 0.5)')
plt.show()