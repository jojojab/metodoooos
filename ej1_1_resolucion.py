import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ej1_lore import calculate_similarity_matrix, pca

def main():
    dataset = pd.read_csv('dataset.csv')
    X = dataset.drop(columns=['Unnamed: 0'])

    plt.figure(figsize=(8, 12))
    plt.imshow(X, interpolation='nearest', aspect='auto')
    plt.title("X")
    plt.colorbar(label='Similarity Value')
    plt.tight_layout()
    plt.show()

    d_values = [2, 6, 10, 106]
    sigma = 1

    Clusters = None

    for d in d_values:
        Z = pca(X, d)
        similarity_matrix_reduced = calculate_similarity_matrix(Z, sigma)
        plot_similarity_matrix(similarity_matrix_reduced, f'Similarity Matrix for d={d}')

        if d == 2:
            Clusters = Z

    plot_clusters(Clusters)

def plot_similarity_matrix(matrix, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation='nearest')
    plt.title(title)
    plt.colorbar(label='Similarity Value')
    plt.tight_layout()
    plt.show()


def plot_clusters(Z):
    Y = np.loadtxt('y.txt')

    plt.style.use('ggplot')

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=Y, cmap='viridis', edgecolor='k')
    plt.title('Clusters in 2D PCA space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.colorbar(scatter, label='Value of Y')

    plt.tight_layout()
    plt.show()

    plot_clusters_3d(Z)


def plot_clusters_3d(Z):
    Y = np.loadtxt('y.txt')

    plt.style.use('ggplot')

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z[:, 0], Z[:, 1], Y, c=Y, cmap='viridis', edgecolor='k')
    ax.set_title('Clusters in 3D PCA space')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Value of Y')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
