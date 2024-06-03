from matplotlib import pyplot as plt
from ej2_lore import images_to_matrix, image_reconstructed


def plot_reconstruction(a_d, cant_images, d):
    fig, axs = plt.subplots(4, 5)
    for i in range(cant_images):
        row = i // 5
        col = i % 5
        axs[row, col].imshow(a_d[i], cmap='gray')
        axs[row, col].axis('off')
    #     delete the empty axs
    fig.delaxes(axs[3, 4])
    fig.suptitle('Images Reconstructed for d = ' + str(d))
    plt.show()
    plt.close()


def main():
    cant_images = 19
    images_directory = 'images1'
    dim = [2, 3, 4, 5, 8, 10, 15, 17, 19]

    all_image_vectors = images_to_matrix(images_directory, cant_images)
    for d in dim:
        u_d, s_d, vt_d, a_d = image_reconstructed(all_image_vectors, d)
        plot_reconstruction(a_d.reshape(cant_images, 28, 28), cant_images, d)


if __name__ == '__main__':
    main()
