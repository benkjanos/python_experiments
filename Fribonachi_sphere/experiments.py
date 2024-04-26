import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sample_spherical(npoints, ndim=3, radius=1):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return radius * vec

def plot_3d_spherical(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = points[0]
    y = points[1]
    z = points[2]
    c = z  # Color by z-coordinate

    ax.scatter(x, y, z, c=c, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Scatter Plot of Spherical Samples')
    plt.show()

# Example usage:
npoints = 100
ndim = 3
radius = 1.0  # Change the radius as per your requirement
points = sample_spherical(npoints, ndim, radius)
#print(points)
plot_3d_spherical(points)