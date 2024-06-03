from matplotlib import pyplot as plt

from ej2_lore import calculate_similarity_matrix
import numpy as np
from PIL import Image
import os

def image_to_vector(image_path):
    image = Image.open(image_path)
    image_matrix = np.array(image)
    return image_matrix.flatten().tolist()

def images_to_matrix(images_directory, cant_images):
    all_image_vectors = []
    for i in range(cant_images):  # Adjust range according to your image naming convention
        image_filename = f'img{i:02d}.jpeg'  # Format the filename
        image_path = os.path.join(images_directory, image_filename)
        if os.path.exists(image_path):
            image_vector = image_to_vector(image_path)
            all_image_vectors.append(image_vector)
    return all_image_vectors

def reconstruct_images(data, d):
    U, S, Vt = np.linalg.svd(data, full_matrices=False)
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]
    A_d = U_d @ S_d @ Vt_d
    reduced_data = U_d @ S_d
    return U_d, S_d, Vt_d, reduced_data

def compute_similarity(reduced_data):
    ##CENTRO ACA O ARRIBA?
    # reduced_data = reduced_data - np.mean(reduced_data, axis=0)
    dot_product = np.dot(reduced_data, reduced_data.T)
    norms = np.linalg.norm(reduced_data, axis=1)
    similarity_matrix = dot_product / np.outer(norms, norms)
    return similarity_matrix

# Calcular la similaridad entre pares de imágenes para diferentes valores de d
def analyze_similarity(data, d_values):
    plt.figure(figsize=(15, 10))
    for d in d_values:
        U, S, VT, reduced_data = reconstruct_images(data, d)
        similarity_matrix = compute_similarity(reduced_data)
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