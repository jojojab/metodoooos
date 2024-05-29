import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ej1_lore import calculate_similarity_matrix


def main():

    # Load the dataset
    dataset = pd.read_csv('dataset.csv')

    # Standardize the data
    X = (dataset - dataset.mean()) / dataset.std()

    # Perform SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    #PCA
    d_values = [2, 6, 10]
    sigma = 1.0

    for d in d_values:
        Z = U[:, :d] @ np.diag(S[:d])
        similarity_matrix_reduced = calculate_similarity_matrix(Z, sigma)
        plot_similarity_matrix(similarity_matrix_reduced, f'Similarity Matrix for d={d}')

def plot_similarity_matrix(matrix, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation='nearest')
    plt.title(title)
    plt.colorbar(label='Similarity Value')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()


