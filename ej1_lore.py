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
    # Centrar los datos
    X_centered = X - np.mean(X, axis=0)
    # Calcular la matriz de covarianza
    covariance_matrix = np.cov(X_centered, rowvar=False)
    # Calcular los valores y vectores propios
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # Ordenar los vectores propios por los valores propios en orden descendente
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    # Seleccionar los primeros n_components vectores propios
    eigenvectors_subset = sorted_eigenvectors[:, :n_components]
    # Transformar los datos
    X_reduced = np.dot(X_centered, eigenvectors_subset)
    return X_reduced
