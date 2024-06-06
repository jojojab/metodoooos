from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import PercentFormatter
from ej2_lore import norma_de_frobenius, image_to_vector, image_reconstructed


def image_to_matrix(image_path, d):
    image_filename = f'img{d:02d}.jpeg'
    image_full_path = os.path.join(image_path, image_filename)
    if os.path.exists(image_full_path):
        image = Image.open(image_full_path).convert('L')
        image_matrix = np.array(image)
        return image_matrix
    else:
        raise FileNotFoundError(f"No se encontró el archivo {image_full_path}")


def images_to_matrix(images_directory, cant_images):
    all_image_vectors = []
    for i in range(cant_images):
        image_filename = f'img{i:02d}.jpeg'
        image_path = os.path.join(images_directory, image_filename)
        if os.path.exists(image_path):
            image_vector = image_to_vector(image_path)
            all_image_vectors.append(image_vector)
        else:
            raise FileNotFoundError(f"No se encontró el archivo {image_path}")
    return np.array(all_image_vectors)


def plot_commited_errors(errores):
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(errores) + 1), [error * 100 for error in errores], color='blue')
    plt.scatter(range(1, len(errores) + 1), [error * 100 for error in errores], color='blue')
    plt.axhline(y=10, color='r', linestyle='--')
    plt.xlabel('Número de dimensiones')
    plt.ylabel('Error')
    plt.xlim(0, len(errores) + 1)
    plt.ylim(-5, 100)
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.title('Error cometido en función de las dimensiones')
    plt.xticks(range(len(errores) + 1))
    plt.yticks(range(0, 101, 10))
    plt.show()


def plot_commited_errors_for_images(errores):
    plt.bar(range(0, len(errores)), [error * 100 for error in errores])
    plt.xlabel('Número de imagen')
    plt.ylabel('Error')
    plt.xlim(-1, len(errores))
    plt.ylim(0, 100)
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.title('Error cometido en función de las imágenes')
    plt.xticks(range(len(errores)))
    plt.show()


def change_base(a, vt):
    return a @ vt.T @ vt

def plot_reconstruction(cant_images, images_directory, a_d, vt2, errors2):
    fig, axs = plt.subplots(4, 5)
    for i in range(cant_images):
        reshaped_reconstructed_image = change_base(a_d[i], vt2).reshape(28, 28)
        original_image = image_to_matrix(images_directory, i)
        errors2.append(norma_de_frobenius(original_image, reshaped_reconstructed_image))
        row = i // 5
        col = i % 5
        axs[row, col].imshow(reshaped_reconstructed_image, cmap='gray')
        axs[row, col].axis('off')
        axs[row, col].set_title(f'Imagen {i+1}')
    #     make the titles smaller
    for ax in axs.flat:
        ax.title.set_fontsize(8)
    # delete empty axes
    fig.delaxes(axs[3, 4])
    fig.suptitle('Imagenes Reconstruidas')
    plt.show()
    plt.close()


def plot_images_dataset2(data):
    fig, axs = plt.subplots(2, 4)
    for i in range(8):
        row = i // 4
        col = i % 4
        axs[row, col].imshow(data[i].reshape(28, 28), cmap='gray')
        axs[row, col].axis('off')
        axs[row, col].set_title(f'Imagen {i+1}')
    fig.suptitle('Imagenes dataset 2')
    plt.show()
    plt.close()

def plot_autovectores(vt):
    fig, axs = plt.subplots(2, 4)
    for i in range(8):
        row = i // 4
        col = i % 4
        axs[row, col].imshow(vt[i].reshape(28, 28), cmap='gray')
        axs[row, col].axis('off')
        axs[row, col].set_title(f'Autovector {i+1}')
        axs[row, col].title.set_fontsize(8)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.suptitle('Autovectores')
    plt.show()
    plt.close()



def main():
    plt.style.use('ggplot')
    cant_images = [19, 8]
    images_directory = ['images1', 'datasets_imgs_02']
    all_image_vectors = images_to_matrix(images_directory[0], cant_images[0])
    all_image_vectors2 = images_to_matrix(images_directory[1], cant_images[1])
    errors = []
    errors2 = []
    dim = 0

    plot_images_dataset2(all_image_vectors2)

    for d in range(1, 11):
        u2, s2, vt2, a_d2 = image_reconstructed(all_image_vectors2, d)
        errors.append(norma_de_frobenius(all_image_vectors2, a_d2))
        if (errors[-1] < 0.1) & (dim == 0):
            dim = d

    plot_autovectores(vt2)

    u, s, vt, a_d = image_reconstructed(all_image_vectors, dim)
    plot_reconstruction(cant_images[0], images_directory[0], a_d, vt2, errors2)

    plot_commited_errors(errors)
    plot_commited_errors_for_images(errors2)



if __name__ == '__main__':
    main()
