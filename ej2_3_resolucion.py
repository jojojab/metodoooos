from matplotlib import pyplot as plt
import numpy as np
from ej2_lore import images_to_matrix, image_reconstructed


def compute_similarity(reduced_data):
    dot_product = np.dot(reduced_data, reduced_data.T)
    norms = np.linalg.norm(reduced_data, axis=1)
    similarity_matrix = dot_product / np.outer(norms, norms)
    return similarity_matrix


def analyze_similarity(data, d_values):
    fig, axs = plt.subplots(2, 3)
    for d in range(len(d_values)):
        row = d // 3
        col = d % 3
        u, s, vt, a_d = image_reconstructed(data, d_values[d])
        similarity_matrix = compute_similarity(a_d)
        im = axs[row, col].imshow(similarity_matrix, cmap='hot', interpolation='nearest')
        axs[row, col].set_title(f'd = {d_values[d]}')
        axs[row, col].set_xticks(np.arange(0, 19, 1))
        axs[row, col].set_yticks(np.arange(0, 19, 1))
    fig.delaxes(axs[1, 2])
    fig.suptitle('Matrices de similaridad para distintas dimensiones')
    fig.colorbar(im, ax=axs)
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

        # axs[i].imshow(a_d[images_index[i]].reshape(28, 28), cmap='gray')
        # axs[i].axis('off')
        # axs[i].set_title(f'Imagen {images_index[i]+1}')
    fig.suptitle(f'Imagenes similares en dimension {d}')
    plt.show()



def main():
    cant_images = 19
    images_directory = 'images1'
    dim = [15, 10, 5, 3, 2]

    all_image_vectors = images_to_matrix(images_directory, cant_images)
    # plot_similarity(all_image_vectors, 19)
    
    # analyze_similarity(all_image_vectors, dim)
    
    plot_similar_images(all_image_vectors, [1, 17], 15)
    plot_similar_images2(all_image_vectors, [0, 6, 7, 9, 14, 16], 10)


if __name__ == '__main__':
    main()
