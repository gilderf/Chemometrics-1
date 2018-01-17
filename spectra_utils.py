# preprocessing
import numpy as np


def SNV(X, offset=0):
    """
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




