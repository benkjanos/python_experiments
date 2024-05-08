import numpy as np

def generate_covariance_matrix(n, min_corr=-0.5, max_corr=0.5):
    """
    Generate a random covariance matrix with realistic values.

    Parameters:
        n (int): Number of assets.
        min_corr (float): Minimum correlation value.
        max_corr (float): Maximum correlation value.

    Returns:
        numpy.ndarray: Covariance matrix.
    """
    # Generate a random correlation matrix
    corr = np.random.uniform(min_corr, max_corr, size=(n, n))
    np.fill_diagonal(corr, 1)

    # Make the matrix symmetric
    corr = (corr + corr.T) / 2

    # Generate a random positive semi-definite matrix
    _, s, vh = np.linalg.svd(corr)
    s = np.diag(s)
    cov_matrix = np.dot(np.dot(vh.T, s), vh)

    return cov_matrix

# Example usage
n = 5  # Number of assets
cov_matrix = generate_covariance_matrix(n)
print("Random Covariance Matrix:")
print(cov_matrix)