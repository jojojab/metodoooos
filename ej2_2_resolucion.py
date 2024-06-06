from matplotlib import pyplot as plt
from ej2_lore import images_to_matrix, image_reconstructed


def plot_reconstruction(a_d, cant_images, d):
    fig, axs = plt.subplots(4, 5)
    for i in range(cant_images):
        row = i // 5
        col = i % 5
        axs[row, col].imshow(a_d[i].reshape(28, 28), cmap='gray')
        axs[row, col].axis('off')
    #     delete the empty axs
    fig.delaxes(axs[3, 4])
    fig.suptitle('Images Reconstructed for d = ' + str(d))
    plt.show()
    plt.close()

def plot_reconstruction_all_dim(all_image_vectors, cant_images, dim):
    fig, axs = plt.subplots(cant_images, len(dim))
    for i in range(cant_images):
        for j in range(len(dim)):
            row = i
            col = j
            u_d, s_d, vt_d, a_d = image_reconstructed(all_image_vectors, dim[j])
            axs[row, col].imshow(a_d[i].reshape(28, 28), cmap='gray')
            axs[row, col].axis('off')
            if i == 0:
                axs[row, col].set_title(f'Dim = {dim[j]}')
    fig.suptitle('Reconstruccion de imagenes para distintas dimensiones')
    plt.show()
    plt.close()

def main():
    cant_images = 19
    cant_images1 = 4
    images_directory = 'images1'
    dim = [15, 10, 5, 3, 2]

    all_image_vectors = images_to_matrix(images_directory, cant_images)
    plot_reconstruction_all_dim(all_image_vectors, cant_images1, dim)
    u_d, s_d, vt_d, a_d = image_reconstructed(all_image_vectors, 2)
    plot_reconstruction(a_d, cant_images, 2)
    u_d, s_d, vt_d, a_d = image_reconstructed(all_image_vectors, 19)
    plot_reconstruction(a_d, cant_images, 19)


if __name__ == '__main__':
    main()
