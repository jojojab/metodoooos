import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    dataset = pd.read_csv('dataset.csv')

    X = dataset.drop(columns=['Unnamed: 0'])

    # Realizar SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    plt.figure(figsize=(10, 6))
    plt.bar([i for i in range(len(S))], S)
    plt.xlabel('Dimensiones originales')
    plt.ylabel('Importancia')
    plt.title('Importancia de cada dimensión original')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
