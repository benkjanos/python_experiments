import numpy as np


def fibonacci_sphere(samples=1, randomize=False):
    """
    Generates uniformly distributed points on a unit sphere using Fibonacci sphere sampling.

    Parameters:
        samples (int): Number of points to generate.
        randomize (bool): If True, randomizes the order of points.

    Returns:
        numpy.array: Array of shape (samples, 3) containing the sampled points.
    """
    points = np.zeros((samples, 3))
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points[i] = [x, y, z]

    if randomize:
        np.random.shuffle(points)

    return points


pl = fibonacci_sphere(samples=30, randomize=False)

print(" r = ", str(pl))
