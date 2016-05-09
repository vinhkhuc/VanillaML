"""
Misc utility
"""
import numpy as np


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def sign(x):
    return 1 if x >= 0 else -1


def get_penalty(X, factor, type):
    """ Get penalty for the input ndarray.

    Args:
        X (ndarray): input data, shape N x 1
        factor (float): penalty factor
        type (str): penalty type

    Returns:
        ndarray: penalty values

    """
    assert X.ndim == 1, "The input array must be one-dimensional."
    assert factor > 0, "The penalty factor must be positive."

    if type == 'l1':
        return factor * np.sum(np.abs(X))
    elif type == 'l2':
        return factor * np.inner(X, X)
    else:
        raise Exception("The penalty '%s' is not supported!" % type)


def sign_prediction(y):
    """
    Maps {0, 1} to {-1, 1}.
    """
    return 2 * y - 1


def unsign_prediction(y):
    """
    Maps {-1, 1} to {0, 1}.
    """
    return (y + 1) / 2
