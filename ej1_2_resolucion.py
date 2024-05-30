import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Cargar el dataset
    dataset = pd.read_csv('dataset.csv')

    # Estandarizar los datos
    X = (dataset - dataset.mean()) / dataset.std()

    # Realizar SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    plt.figure(figsize=(10, 6))
    plt.bar([i for i in range(len(S))],S)
    plt.xlabel('Dimensiones originales')
    plt.ylabel('Importancia')
    plt.title('Importancia de cada dimensión original')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
