import numpy as np
from PIL import Image
import os


def image_to_vector(image_path, flattened=True):
    image = Image.open(image_path)
    image_matrix = np.array(image)
    if not flattened:
        return image_matrix
    return image_matrix.flatten().tolist()


def images_to_matrix(images_directory, cant_images, flattened=True):
    all_image_vectors = []
    for i in range(cant_images):  # Adjust range according to your image naming convention
        image_filename = f'img{i:02d}.jpeg'  # Format the filename
        image_path = os.path.join(images_directory, image_filename)
        if os.path.exists(image_path):
            image_vector = image_to_vector(image_path, flattened)
            all_image_vectors.append(image_vector)
    return all_image_vectors


def image_reconstructed(data, d):
    u, s, vt = np.linalg.svd(data, full_matrices=False)
    u_d = u[:, :d]
    s_d = np.diag(s[:d])
    vt_d = vt[:d, :]
    a_d = u_d @ s_d @ vt_d
    return u_d, s_d, vt_d, a_d


def norma_de_frobenius(a, a_d):
    return np.linalg.norm(a - a_d, 'fro') / np.linalg.norm(a, 'fro')
