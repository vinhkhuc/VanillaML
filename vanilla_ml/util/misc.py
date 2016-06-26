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
        w (ndarray): input data, shape K x P
        factor (float): penalty factor
        penalty (str): penalty type

    Returns:
        ndarray: penalty values

    """
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


def one_hot(y, n_classes=None):
    """ Convert an 1D array to one-hot 2D array (using one-liner trick).

    Args:
        y (ndarray): uint array, shape N
        n_classes (Optional[int]): number of classes.

    Returns:
        ndarray: one-hot 2D array, shape N x K

    """
    if n_classes is None:
        n_classes = len(np.unique(y))
    return np.eye(n_classes)[y]


def softmax(X):
    """ Compute softmax (based on the formulas 3.70 and 8.33 in Kevin Murphy's book).

    Args:
        X (ndarray): array, shape N x K

    Returns:
        ndarray: softmax, shape N x 1.

    """
    log_sum_exp_X = log_sum_exp(X)
    return np.exp(X - log_sum_exp_X[:, None])


def log_sum_exp(X):
    """ Compute log of sum of exps.
    Using the log-sum-exp trick as shown in the formula 3.74 in Kevin Murphy's book.

    Args:
        X (ndarray): array, shape N x K

    Returns:
        ndarray: log-sum-exp results, shape N x 1.

    """
    max_X = X.max(axis=1)
    return max_X + np.log(np.sum(np.exp(X - max_X[:, None]), axis=1))


# def array_equals(a, b, tol):
#     diff = np.sqrt(np.sum(np.square(a - b)))
#     return diff <= tol
