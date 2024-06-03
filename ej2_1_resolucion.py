import numpy as np
import matplotlib.pyplot as plt
from ej2_lore import images_to_matrix


def plot_matrix_u(u):
    plt.figure(figsize=(10, 8))
    plt.imshow(u)
    plt.title('Matrix U')
    plt.colorbar()
    plt.show()


def singular_values_bar_plot(s):
    plt.figure(figsize=(10, 8))
    plt.bar(np.arange(len(s)), s)
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value')
    plt.title('Singular Values Bar Plot')
    plt.grid()
    plt.show()


def plot_matrix_vt(vt):
    plt.figure(figsize=(10, 8))
    plt.imshow(vt)
    plt.title('Matrix Vt')
    plt.colorbar()
    plt.show()


def main():
    cant_images = 19
    images_directory = 'images1'

    all_image_vectors = images_to_matrix(images_directory, cant_images)

    u, s, vt = np.linalg.svd(all_image_vectors, full_matrices=False)

    plot_matrix_u(u)
    singular_values_bar_plot(s)
    plot_matrix_vt(vt)


if __name__ == '__main__':
    main()
