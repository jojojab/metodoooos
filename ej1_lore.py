import numpy as np


def calculate_similarity_matrix(X, sigma, chunk_size=100):
    n = X.shape[0]
    similarity_matrix = np.zeros((n, n))
    for i in range(0, n, chunk_size):
        for j in range(0, n, chunk_size):
            X_chunk = X[i:i + chunk_size]
            Y_chunk = X[j:j + chunk_size]
            pairwise_sq_dists = np.sum((X_chunk[:, np.newaxis] - Y_chunk[np.newaxis, :]) ** 2, axis=-1)
            similarity_matrix[i:i + chunk_size, j:j + chunk_size] = np.exp(-pairwise_sq_dists / (2 * sigma ** 2))
    return similarity_matrix


def pca(X, n_components):
    X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)

    covariance_matrix = np.cov(X_standardized, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    components = sorted_eigenvectors[:, :n_components]

    X_reduced = np.dot(X_standardized, components)

    return X_reduced
