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


def main():
    plt.style.use('ggplot')
    cant_images = [8, 19]
    images_directory = ['datasets_imgs_02', 'images1']
    all_image_vectors = images_to_matrix(images_directory[0], cant_images[0])
    errors = []
    errors2 = []
    dim = 0

    for d in range(1, 11):
        mayor_error = -1
        dim = d
        for i in range(cant_images[0]):
            u, s, vt, a_d = image_reconstructed(all_image_vectors, d)
            reshaped_reconstructed_image = np.reshape(a_d[i], (28, 28))
            original_image = image_to_matrix(images_directory[0], i)
            error = norma_de_frobenius(original_image, reshaped_reconstructed_image)
            if error > mayor_error:
                mayor_error = error
        errors.append(mayor_error)

    all_image_vectors2 = images_to_matrix(images_directory[1], cant_images[1])
    u2, s2, vt2, a_d2 = image_reconstructed(all_image_vectors2, dim)
    for i in range(cant_images[1]):
        reshaped_reconstructed_image = change_base(a_d2[i], vt).reshape(28, 28)
        original_image = image_to_matrix(images_directory[1], i)
        errors2.append(norma_de_frobenius(original_image, reshaped_reconstructed_image))

    plot_commited_errors(errors)
    plot_commited_errors_for_images(errors2)


if __name__ == '__main__':
    main()
