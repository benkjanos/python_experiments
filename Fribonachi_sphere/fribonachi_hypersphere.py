import numpy as np


def fibonacci_hypersphere(samples, dimensions, randomize=False):
    """
    Generates uniformly distributed points on a hypersphere using Fibonacci sphere sampling.

    Parameters:
        samples (int): Number of points to generate.
        dimensions (int): Number of dimensions.
        randomize (bool): If True, randomizes the order of points.

    Returns:
        numpy.array: Array of shape (samples, dimensions) containing the sampled points.
    """
    points = np.zeros((samples, dimensions))
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment
        for d in range(dimensions - 1):
            theta = phi * i * (d + 1)  # golden angle increment for each dimension
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points[i, d] = x
            points[i, -1] = y

    if randomize:
        np.random.shuffle(points)

    return points


# Example usage:
n_points = 100
dimensions = 4
points = fibonacci_hypersphere(n_points, dimensions)
print(f"Sampled points on a unit {dimensions}-dimensional hypersphere with Fibonacci sphere sampling:")
print(points)