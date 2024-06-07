from matplotlib import pyplot as plt
import numpy as np
from ej2_lore import images_to_matrix, image_reconstructed

def compute_similarity(reconstructed_data):
    num_images = reconstructed_data.shape[0]
    similarity_matrix = np.zeros((num_images, num_images))
    for i in range(num_images):
        for j in range(i, num_images):
            vector1 = reconstructed_data[i].reshape(28, 28)
            vector2 = reconstructed_data[j].reshape(28, 28)
            difference = vector1 - vector2
            error = np.linalg.norm(difference, 'fro') / np.linalg.norm(vector1, 'fro')
            similarity_matrix[i, j] = similarity_matrix[j, i] = error
    return similarity_matrix


def analyze_similarity(data, d_values):
    plt.figure(figsize=(15, 10))
    for d in d_values:
        U, S, VT, a_d = image_reconstructed(data, d)
        similarity_matrix = compute_similarity(a_d)
        plt.subplot(1, 5, d_values.index(d) + 1)
        plt.imshow(similarity_matrix, cmap='hot_r', interpolation='nearest')
        plt.title(f'Matriz de similaridad para d = {d}')
        plt.colorbar(fraction=0.046, pad=0.04)
    plt.suptitle('Matrices de Similaridad', fontsize=20, weight='bold')
    plt.tight_layout()
    plt.show()

def plot_similarity(data, d):
    u, s, vt, a_d = image_reconstructed(data, d)
    similarity_matrix = compute_similarity(a_d)
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.title(f'Matriz de similaridad para d = {d}')
    plt.colorbar()
    plt.xticks(np.arange(0, 19, 1))
    plt.yticks(np.arange(0, 19, 1))
    plt.show()

def plot_similar_images(data, images_index, d):
    u, s, vt, a_d = image_reconstructed(data, d)
    fig, axs = plt.subplots(1, len(images_index))
    for i in range(len(images_index)):
        axs[i].imshow(a_d[images_index[i]].reshape(28, 28), cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f'Imagen {images_index[i]+1}')
    fig.suptitle(f'Imagenes similares en dimension {d}')
    plt.show()

def plot_similar_images2(data, images_index, d):
    u, s, vt, a_d = image_reconstructed(data, d)
    fig, axs = plt.subplots(2, len(images_index)//2)
    for i in range(len(images_index)):
        row = i // 3
        col = i % 3
        axs[row, col].imshow(a_d[images_index[i]].reshape(28, 28), cmap='gray')
        axs[row, col].axis('off')
        axs[row, col].set_title(f'Imagen {images_index[i]+1}')
    fig.suptitle(f'Imagenes similares en dimension {d}')
    plt.show()



def main():
    cant_images = 19
    images_directory = 'images1'
    dim = [15, 10, 5, 3, 2]

    all_image_vectors = images_to_matrix(images_directory, cant_images)
    plot_similarity(all_image_vectors, 19)

    analyze_similarity(all_image_vectors, dim)
    
    plot_similar_images(all_image_vectors, [1, 17], 15)
    plot_similar_images2(all_image_vectors, [0, 6, 7, 9, 14, 16], 10)


if __name__ == '__main__':
    main()
