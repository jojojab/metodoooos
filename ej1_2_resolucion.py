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

    # Calcular la importancia de cada dimensión original
    dimension_importance = calculate_dimension_importance(Vt, S)

    # Graficar la importancia de cada dimensión original
    plot_dimension_importance(dimension_importance, dataset.columns)


def calculate_dimension_importance(Vt, S):
    # Multiplicar los valores absolutos de los vectores propios por los correspondientes valores singulares
    importance = np.sum(np.abs(Vt.T) * S, axis=1)
    return importance


def plot_dimension_importance(importance, feature_names):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance, tick_label=feature_names)
    plt.xlabel('Dimensiones originales')
    plt.ylabel('Importancia')
    plt.title('Importancia de cada dimensión original')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
