# preprocessing
import numpy as np


def SNV(X, offset=0):
    """
    样本（一行，针对观测）的方差归一化
    Standard Normal Variate
    :param X: 2d numpy array, a row is a sample, columns ara features
    :param offset:
    :return:
    """
    w = X.std(axis=1) + offset
    X = X/w[:, np.newaxis]
    return X


def MSC(X, reference=None):
    """
    Multiplicative Scatter Correction
    :param X: X=np.random.randn(2,10)
    :param reference:
    :return:
    """
    mX = X.mean(axis=1, keepdims=True)
    if reference is None:
        reference = X.mean(axis=0)
    X -= mX
    reference -= reference.mean()
    b = reference[np.newaxis, :].dot(X.T)/(reference.dot(reference))
    X = X/b.T+reference.mean()
    return X


def first_derivative(X):
    # 一阶导数

    d1_X = derivative(X, degree=1)
    return d1_X


def derivative(X, degree=1, axis=0):
    # 导数(列)

    d = np.diff(X, n=degree, axis=axis)
    d1_X = np.vstack([np.nan*np.zeros((degree, X.shape[1])), d])
    return d1_X


def test_SNV_MSC_derivatie():
    """
    test
    :return:
    """
    X = np.random.randn(2, 10)
    MSC(X)
    SNV(X)
    first_derivative(X.T)


if __name__ == '__main__':
    test_SNV_MSC_derivatie()






