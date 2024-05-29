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