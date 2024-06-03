from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.ticker import PercentFormatter

from ej2_lore import norma_de_frobenius

def image_to_matrix(image_path, d):
    image_filename = f'img{d:02d}.jpeg'
    image_full_path = os.path.join(image_path, image_filename)
    if os.path.exists(image_full_path):
        image = Image.open(image_full_path).convert('L')
        image_matrix = np.array(image)
        return image_matrix
    else:
        raise FileNotFoundError(f"No se encontró el archivo {image_full_path}")

def image_to_vector(image_path):
    image = Image.open(image_path).convert('L')
    image_matrix = np.array(image)
    return image_matrix.flatten()

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

def reconstruct_images(data, d):
    U, S, Vt = np.linalg.svd(data, full_matrices=False)
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]
    A_d = U_d @ S_d @ Vt_d
    return U_d, S_d, Vt_d, A_d

def plot_commited_errors(errores):
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(errores) + 1), [error * 100 for error in errores], color = 'blue')
    plt.scatter(range(1, len(errores) + 1), [error * 100 for error in errores], color = 'blue')
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

def change_base(A, Vt):
    return A @ Vt.T @ Vt

def main():
    plt.style.use('ggplot')
    cant_images = 8
    images_directory = 'datasets_imgs_02'
    all_image_vectors = images_to_matrix(images_directory, cant_images)
    errores = []
    dim = 0

    for d in range(1, 11):
        mayor_error = -1
        dim = d
        for i in range(cant_images):
            U, S, VT, A_d = reconstruct_images(all_image_vectors, d)
            reshaped_reconstructed_image = np.reshape(A_d[i], (28, 28))
            original_image = image_to_matrix(images_directory, i)
            error = norma_de_frobenius(original_image, reshaped_reconstructed_image)
            if error > mayor_error:
                mayor_error = error
        errores.append(mayor_error)
        if mayor_error < 0.1:
            print(f'El número mínimo de dimensiones es {d}')

    plot_commited_errors(errores)

    images_directory2 = 'images1'
    cant_images2 = 19
    errores2 = []

    all_image_vectors2 = images_to_matrix(images_directory2, cant_images2)
    U2, S2, VT2, A_d2 = reconstruct_images(all_image_vectors2, dim)
    for i in range(cant_images2):
        reshaped_reconstructed_image = change_base(A_d2[i], VT).reshape(28, 28)
        original_image = image_to_matrix(images_directory2, i)
        errores2.append(norma_de_frobenius(original_image, reshaped_reconstructed_image))

    plot_commited_errors_for_images(errores2)

if __name__ == '__main__':
    main()
