"""
Misc utility
"""
import numpy as np


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def sign(x):
    return 1 if x >= 0 else -1


def get_penalty(w, factor, penalty):
    """ Get penalty for the input ndarray.

    Args:
        w (ndarray): input data, shape N x 1
        factor (float): penalty factor
        penalty (str): penalty type

    Returns:
        ndarray: penalty values

    """
    assert w.ndim == 1, "The input array must be one-dimensional."
    assert factor > 0, "The penalty factor must be positive."

    if penalty == 'l1':
        raise Exception("L1 penalty is not supported yet!")
    elif penalty == 'l2':
        return 2 * factor * w
    else:
        raise Exception("The penalty '%s' is not supported!" % penalty)


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
