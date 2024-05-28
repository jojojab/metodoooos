import numpy as np
import pandas as pd
from scipy.linalg import svd
from sdv.single_table import GaussianCopulaSynthesizer
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
dataset_path = 'dataset.csv'
y_path = 'y.txt'
data = pd.read_csv(dataset_path)
y = pd.read_csv(y_path, header=None).values.flatten()

# Function to compute similarity matrix
def compute_similarity_matrix(X, sigma):
    n = X.shape[0]
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = np.exp(-np.linalg.norm(X[i] - X[j])**2 / (2 * sigma**2))
    return similarity_matrix

# Function to reduce dimensionality using SVD
def reduce_dimensionality(X, d):
    U, S, Vt = svd(X, full_matrices=False)
    X_reduced = np.dot(U[:, :d], np.diag(S[:d]))
    return X_reduced

# Compute similarity matrices for different dimensions
dimensions = [2, 6, 10, data.shape[1]]
sigma = 1.0  # You can adjust sigma as needed

similarity_matrices = {}
for d in dimensions:
    if d < data.shape[1]:
        X_reduced = reduce_dimensionality(data.values, d)
    else:
        X_reduced = data.values
    similarity_matrices[d] = compute_similarity_matrix(X_reduced, sigma)

# Print similarity matrices
for d, sim_matrix in similarity_matrices.items():
    print(f"Similarity matrix for dimension {d}:\n{sim_matrix}\n")

for d, sim_matrix in similarity_matrices.items():
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap='viridis')
    plt.title(f'Similarity Matrix for Dimension {d}')
    plt.show()
