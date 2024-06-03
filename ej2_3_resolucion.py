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

def reconstruct_images(U, S, Vt, d):
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    Vt_d = Vt[:d, :]
    A_d = U_d @ S_d @ Vt_d
    return A_d

def similarity_percentage(similarity_matrix):
    total_sum = 0
    total_count = 0

    for row in similarity_matrix:
        total_sum += sum(row)
        total_count += len(row)

    if total_count == 0:
        return 0  # To handle the case of an empty matrix

    average = total_sum / total_count
    return average

def plot_similarity_percentage(similarity_percentage_matrix, dim):
    for idx, matrix in enumerate(similarity_percentage_matrix):
        plt.figure(figsize=(5, 5))
        plt.imshow(matrix, cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title(f'Dimension: {dim[idx]}')
        plt.show()


def main():
    cant_images = 19
    images_directory = 'images1'
    sigma = 1
    dim = [2, 3, 4, 5, 10, 15, 20, 28]

    all_image_vectors = images_to_matrix(images_directory, cant_images)
    U, S, Vt = np.linalg.svd(all_image_vectors, full_matrices=False)
    similarity_percentage_matrix = []
    sim_percentage_matrix = np.zeros((cant_images, cant_images))
    for i in range(len(dim)):
        for j in range(cant_images):
            for k in range(cant_images):
                A_d = reconstruct_images(U, S, Vt, dim[i])[j].reshape(28, 28)
                B_d = reconstruct_images(U, S, Vt, dim[i])[k].reshape(28, 28)

                similarity_matrix = calculate_similarity_matrix(A_d, B_d, sigma)
                # print(similarity_matrix)
                # print(similarity_matrix.shape)
                sim_percentage_matrix[j, k] = similarity_percentage(similarity_matrix)
        similarity_percentage_matrix.append(sim_percentage_matrix.tolist())
    # print(similarity_percentage_matrix)

    plot_similarity_percentage(similarity_percentage_matrix, dim)

if __name__ == '__main__':
    main()