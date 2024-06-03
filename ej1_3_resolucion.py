import pandas as pd
from ej1_lore import pca
import numpy as np

#  aumentar dimension no mejora la presicion sino que le da mas libertad al modelo, hace Ax-y_i

def main():
    dataset = pd.read_csv('dataset.csv')

    Y = np.loadtxt('y.txt')
    Y -= Y.mean()

    X = dataset.drop(columns=['Unnamed: 0'])

    min = 100000
    best_d = 0
    for i in range(2000):
        newmin = calc_precision(X, Y, i)
        if newmin < min:
            min = newmin
            best_d = i

    print(min)
    print(best_d)


def calc_precision(X, Y, d):

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standardized = (X - X_mean) / X_std

    U, S, Vt = np.linalg.svd(X_standardized, full_matrices=False)

    Ud = U[:, :d]
    Vtd = Vt[:d, :]
    Sd_inv = np.diag(1 / S[:d])

    A_daga = Vtd.T @ Sd_inv @ Ud.T

    Y = Y.reshape(-1, 1)

    X_moño = A_daga @ Y

    residuals = X_standardized @ X_moño - Y
    frobenius_norm = np.linalg.norm(residuals, 'fro')

    return frobenius_norm


if __name__ == '__main__':
    main()
