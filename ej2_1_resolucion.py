from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

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

def plot_matrix_u(U):
    plt.figure(figsize=(10, 8))
    plt.imshow(U)
    plt.title('Matrix U')
    plt.colorbar()
    plt.show()

def singular_values_bar_plot(S):
    plt.figure(figsize=(10, 8))
    plt.bar(np.arange(len(S)), S)
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value')
    plt.title('Singular Values Bar Plot')
    plt.grid()
    plt.show()

def plot_matrix_vt(Vt):
    plt.figure(figsize=(10, 8))
    plt.imshow(Vt)
    plt.title('Matrix Vt')
    plt.colorbar()
    plt.show()

def main():
    plt.style.use('ggplot')
    cant_images = 19
    images_directory = 'images1'

    all_image_vectors = images_to_matrix(images_directory, cant_images)

    U, S, Vt = np.linalg.svd(all_image_vectors, full_matrices=False)

    plot_matrix_u(U)
    singular_values_bar_plot(S)
    plot_matrix_vt(Vt)



if __name__ == '__main__':
    main()