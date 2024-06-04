import pandas as pd
import matplotlib.pyplot as plt
from ej1_lore import calculate_similarity_matrix
from ej1_lore import pca


def main():
    plt.style.use('ggplot')

    dataset = pd.read_csv('dataset.csv')

    X = dataset.drop(columns=['Unnamed: 0'])

    plt.figure(figsize=(8, 12))
    plt.imshow(X, interpolation='nearest', aspect='auto')
    plt.title("X")
    plt.colorbar(label='Similarity Value')
    plt.tight_layout()
    plt.show()

    # PCA
    d_values = [2, 6, 10]
    sigma = 1.0

    for d in d_values:
        Z = pca(X, d)
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


