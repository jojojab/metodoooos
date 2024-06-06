import pandas as pd
import matplotlib.pyplot as plt
from ej1_lore import *


def main():

    plt.style.use('ggplot')

    dataset = pd.read_csv('dataset.csv')
    X = dataset.drop(columns=['Unnamed: 0'])

    X_standart = standarize(X)

    U, S, VT = np.linalg.svd(X_standart)

    print("Primeras 10 valores singulares mas importantes")
    print(S[:10])

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(S)), S, color='blue', alpha=0.7)
    plt.xlabel('Index', fontsize=16)
    plt.ylabel('Singular Value', fontsize=16)
    plt.title('Singular Values', fontsize=25)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    eigenvector_plot(ax1, VT[0], 'red')
    eigenvector_plot(ax2, VT[1], 'blue')

    plt.show()


def eigenvector_plot(ax, V, color):
    ax.bar(range(len(V)), np.abs(V), color=color, alpha=0.7)
    ax.set_xlabel('Component', fontsize=16.5)
    ax.set_ylabel('First Eigenvector', fontsize=16.5)
    ax.set_title('Second Eigenvector', fontsize=25)
    ax.grid(True)


if __name__ == '__main__':
    main()