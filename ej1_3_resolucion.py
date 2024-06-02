import pandas as pd
from ej1_lore import pca
import numpy as np


# y restar media de y para alinear con el plano
#  aumentar dimension no mejora la presicion sino que le da mas libertad al modelo, hace Ax-y_i

def main():
    dataset = pd.read_csv('dataset.csv')
    Y = pd.read_csv('y.txt')

    X = dataset.drop(columns=['Unnamed: 0'])


def calc_presition(X, Y, d):

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standardized = (X - X_mean) / X_std

    U, S, Vt = np.linalg.svd(X_standardized)

    A_daga = U.T @ S.inv() @ Vt.T



if __name__ == '__main__':
    main()
