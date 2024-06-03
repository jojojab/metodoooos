import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ej1_lore import standarize

def main():
    dataset = pd.read_csv('dataset.csv')

    Y = np.loadtxt('y.txt')
    Y -= Y.mean()

    X = standarize(dataset.drop(columns=['Unnamed: 0']))

    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    max_d = len(U[0])

    Ud_list = [U[:, :d] for d in range(1, max_d + 1)]
    Vtd_list = [Vt[:d, :] for d in range(1, max_d + 1)]

    epsilon = np.max(S) * 1e-10 * 2
    Sd_inv_list = [np.diag(1 / (S[:d] + epsilon * (S[:d] < epsilon))) for d in range(1, max_d + 1)]

    min_frobenius_norm = float('inf')
    best_d = 0
    errors = []

    for i in range(max_d):
        frobenius_norm = calc_precision(
            X, Y, Ud_list[i], Vtd_list[i], Sd_inv_list[i])

        errors.append(frobenius_norm)

        if frobenius_norm < min_frobenius_norm:
            min_frobenius_norm = frobenius_norm
            best_d = i + 1

    print(f"La mejor dimensión es {best_d} con un error de {min_frobenius_norm}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_d + 1), errors)
    plt.xlim(0, max_d + 1)
    plt.ylim(min(errors) - 10, max(errors) + 10)
    plt.xlabel('Dimensión')
    plt.ylabel('Error en norma 2')
    plt.title('Error entre normas en función de la dimensión')
    plt.grid(True)
    plt.show()

def calc_precision(X_standardized, Y, Ud, Vtd, Sd_inv):
    A_daga = Vtd.T @ Sd_inv @ Ud.T
    X_moño = A_daga @ Y

    residuals = X_standardized @ X_moño - Y

    frobenius_norm = np.linalg.norm(residuals)

    return frobenius_norm


if __name__ == '__main__':
    main()
