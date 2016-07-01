import numpy as np


def check_range(x, left, right):
    """ Check if all elements of the specified array are in the range [left, right]

    Args:
        x (ndarray): shape N
        left (float):
        right (float):

    Returns:
        bool:
    """
    return np.all((x >= left) | (x <= right))


def check_binary(x):
    return np.all((x == 0) | (x == 1))
