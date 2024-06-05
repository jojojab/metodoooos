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
    plt.xlabel('Indice Valores Singulares')
    plt.ylabel('Valores')
    plt.title('Valores Singulares')
    plt.xticks(np.arange(len(s)))
    plt.grid()
    plt.show()


def plot_matrix_vt(vt):
    plt.figure(figsize=(10, 8))
    plt.imshow(vt)
    plt.title('Matrix Vt')
    plt.colorbar()
    plt.show()

def plot_autovectores(vt):
    fig, axs = plt.subplots(4, 5)
    for i in range(19):
        row = i // 5
        col = i % 5
        axs[row, col].imshow(vt[i].reshape(28, 28), cmap='gray')
        axs[row, col].axis('off')
        axs[row, col].set_title(f'Autovector {i+1}')
        axs[row, col].title.set_fontsize(8)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        #     delete the empty axs
    fig.delaxes(axs[3, 4])
    fig.suptitle('Autovectores')
    plt.show()
    plt.close()

def plot_original_images(images_directory, cant_images):
    fig, axs = plt.subplots(4, 5)
    for i in range(cant_images):
        row = i // 5
        col = i % 5
        image = plt.imread(f'{images_directory}/img{i:02d}.jpeg')
        axs[row, col].imshow(image, cmap='gray')
        axs[row, col].axis('off')
        axs[row, col].set_title(f'Imagen {i+1}')
        axs[row, col].title.set_fontsize(8)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        #     delete the empty axs
    fig.delaxes(axs[3, 4])
    fig.suptitle('Imagenes Originales')
    plt.show()
    plt.close()

def main():
    cant_images = 19
    images_directory = 'images1'

    all_image_vectors = images_to_matrix(images_directory, cant_images)

    u, s, vt = np.linalg.svd(all_image_vectors, full_matrices=False)

    plot_matrix_u(u)
    singular_values_bar_plot(s)
    plot_matrix_vt(vt)
    plot_autovectores(vt)
    plot_original_images(images_directory, cant_images)


if __name__ == '__main__':
    main()
