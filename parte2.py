from PIL import Image
import numpy as np
import os

# Function to convert image to vector
def image_to_vector(image_path):
    image = Image.open(image_path)
    image_matrix = np.array(image)
    return image_matrix.flatten().tolist()

# Directory where your images are stored
images_directory = 'images1'

# List to store all image vectors
all_image_vectors = []

# Iterate through all images from img00.jpeg to img18.jpeg
for i in range(2):  # Adjust range according to your image naming convention
    image_filename = f'img{i:02d}.jpeg'  # Format the filename
    image_path = os.path.join(images_directory, image_filename)
    if os.path.exists(image_path):
        image_vector = image_to_vector(image_path)
        all_image_vectors.append(image_vector)


# print(f"Vector of the image img10.jpeg: {all_image_vectors[0]}")
# print(f"Number of images processed: {len(all_image_vectors)}")
# cant_filas = len(all_image_vectors)
# cant_columnas = len(all_image_vectors[0])


from scipy.linalg import lu_factor, lu_solve

def inverse_matrix_lu(A):
    """
    Invert a matrix A using LU decomposition.

    Parameters:
    - A: numpy array of shape (n, p)

    Returns:
    - A_inv: numpy array, inverse of A
    """
    n, p = A.shape

    if n != p:
        raise ValueError("Matrix A must be square for LU decomposition.")

    # Perform LU decomposition
    P, L, U = lu_factor(A)

    # Initialize inverse of A
    A_inv = np.zeros_like(A, dtype=float)

    # Solve for each column of the identity matrix
    I = np.eye(n)
    for i in range(p):
        col_i = I[:, i]
        A_inv[:, i] = lu_solve((P, L, U), col_i)

    return A_inv

# Example usage:
# Create a sample matrix
# A = np.array([[1, 2, 3],
#               [4, 5, 6]])

A = np.array(all_image_vectors)

# Invert the matrix A
A_inv = inverse_matrix_lu(A)
print("Inverse of A:")
print(A_inv)

# Verify the inverse
identity = np.dot(A, A_inv)
print("Identity matrix (A * A_inv):")
print(identity)
