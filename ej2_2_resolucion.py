import numpy as np
from matplotlib import pyplot as plt
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

def images_reconstructed(U, S, Vt, d, cant_images):
    for d_i in d:
        U_d = U[:, :d_i]
        S_d = np.diag(S[:d_i])
        Vt_d = Vt[:d_i, :]
        A_d = U_d @ S_d @ Vt_d
        A_d = A_d.reshape(cant_images, 28, 28)
        fig, axs = plt.subplots(4, 5)
        for i in range(cant_images):
            row = i // 5
            col = i % 5
            axs[row, col].imshow(A_d[i], cmap='gray')
            axs[row, col].axis('off')
        fig.suptitle(f'Images Reconstructed with d = {d_i}')
        plt.show()
        plt.close()

def main():
    cant_images = 19
    images_directory = 'images1'
    dim = [2, 5, 10, 20, 25, 28, 29]

    all_image_vectors = images_to_matrix(images_directory, cant_images)
    U, S, Vt = np.linalg.svd(all_image_vectors, full_matrices=False)
    images_reconstructed(U, S, Vt, dim, cant_images)

if __name__ == '__main__':
    main()