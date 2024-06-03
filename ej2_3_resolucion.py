from matplotlib import pyplot as plt
import numpy as np
from ej2_lore import images_to_matrix, image_reconstructed


def compute_similarity(reduced_data):
    dot_product = np.dot(reduced_data, reduced_data.T)
    norms = np.linalg.norm(reduced_data, axis=1)
    similarity_matrix = dot_product / np.outer(norms, norms)
    return similarity_matrix


def analyze_similarity(data, d_values):
    plt.figure(figsize=(15, 10))
    for d in d_values:
        u, s, vt, a_d = image_reconstructed(data, d)
        similarity_matrix = compute_similarity(a_d)
        plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
        plt.title(f'Matriz de similaridad para d = {d}')
        plt.colorbar()
        plt.show()


def main():
    cant_images = 19
    images_directory = 'images1'
    dim = [2, 3, 4, 5, 8, 10, 15, 17, 19]

    all_image_vectors = images_to_matrix(images_directory, cant_images)
    analyze_similarity(all_image_vectors, dim)


if __name__ == '__main__':
    main()
