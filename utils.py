__author__ = "Li Tao, ltipchrome@gmail.com"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from .basic import timer, plot_cm


def plot_PCA(X, y=None, pca=None, **kwargs):
    if pca is None:
        pca = PCA(n_components=2).fit(X)

    scores = pca.transform(X)

    _ = plt.figure()
    if y is not None:
        classes = np.unique(y)
        for _class in classes:
            plt.scatter(scores[y==_class, 0], scores[y==_class, 1], **kwargs)
    else:
        plt.scatter(*scores.T, **kwargs)
    return pca


def test_plot_PCA():
    X = np.random.randn(10, 100)
    y = np.random.choice(1, X.shape[0])
    plot_PCA(X, y)
    plt.show()


if __name__ == '__main__':
    test_plot_PCA()
