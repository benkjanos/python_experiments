import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def multivariate_gaussian_pdf(x, mu, cov_matrix):
    """
    Calculate the multivariate Gaussian probability density function (PDF).

    Parameters:
        x (numpy.ndarray): Value(s) at which to evaluate the PDF. Shape (n,) or (m, n) for m samples.
        mu (numpy.ndarray): Mean vector of size n.
        cov_matrix (numpy.ndarray): Covariance matrix of size nxn.

    Returns:
        numpy.ndarray: PDF value(s) for the given x.
    """
    n = len(mu)
    det = np.linalg.det(cov_matrix)
    norm_const = 1.0 / ((2 * np.pi) ** (n / 2) * np.sqrt(det))
    x_mu = x - mu
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    result = np.exp(-0.5 * np.einsum('...k,kl,...l->...', x_mu, inv_cov_matrix, x_mu))
    return norm_const * result

# Example usage
n = 2  # Number of dimensions

# Mean vector
mu = np.array([1.0, -2.0]) #zeros(n)

# Covariance matrix
cov_matrix = np.array([[1, -0.8],
                       [-0.8, 2]])

# Define the range for x and y
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y

# Calculate the PDF values
z = multivariate_gaussian_pdf(pos, mu, cov_matrix)

# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Probability Density')
ax.set_title('3D Surface Plot of Multivariate Gaussian Distribution')
plt.show()