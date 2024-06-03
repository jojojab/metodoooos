import numpy as np


def calculate_similarity_matrix(X, Y, sigma):
    n = X.shape[0]
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pairwise_sq_dists = np.sum((X[i, j] - Y[i, j]) ** 2, axis=-1)
            similarity_matrix[i, j] = np.exp(-pairwise_sq_dists / (2 * sigma ** 2))
    return similarity_matrix


def norma_de_frobenius(X):
    return np.linalg.norm(X, 'fro')
