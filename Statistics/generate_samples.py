import numpy as np

def generate_multivariate_gaussian_samples(mu, cov_matrix, num_samples=1000):
    """
    Generate samples from a multivariate Gaussian distribution.

    Parameters:
        mu (numpy.ndarray): Mean vector of size n.
        cov_matrix (numpy.ndarray): Covariance matrix of size nxn.
        num_samples (int): Number of samples to generate.

    Returns:
        numpy.ndarray: Samples from the multivariate Gaussian distribution of size (num_samples, n).
    """
    return np.random.multivariate_normal(mu, cov_matrix, num_samples)

# Example usage
n = 2  # Number of dimensions
num_samples = 30  # Number of samples to generate

# Mean vector
mu = np.zeros(n)

# Covariance matrix
cov_matrix = np.array([[1, 0.5],
                       [0.5, 2]])

# Generate samples
samples = generate_multivariate_gaussian_samples(mu, cov_matrix, num_samples)

# Print the first few samples
print("Generated Samples:")
print(samples)