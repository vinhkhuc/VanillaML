"""
Utility to compute various distances
"""
import numpy as np


def compute_dist_matrix(X1, X2, distance):
    """ Compute the distance matrix where each element m[i][j] is
    the distance between X1[i] and X2[j].

    Args:
        X1 (ndarray): shape N x P.
        X2 (ndarray): shape M x P.
        distance (str): distance type.

    Returns:
        ndarray: distance matrix, shape N x M.

    """
    N, M = X1.shape[0], X2.shape[0]
    dist_matrix = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            dist_matrix[i][j] = dist(X1[i], X2[j], distance=distance)
    return dist_matrix


def dist(x1, x2, distance):
    """ Get distance between two data points.

    Args:
        x1 (ndarray): data point, shape P
        x2 (ndarray): data point, shape P
        distance (str): distance type

    Returns:
        float: distance between the given two data points
    """
    if distance == 'l2':
        return np.sqrt(np.sum(np.square(x1 - x2)))
    elif distance == 'squared_l2':
        return np.sum(np.square(x1 - x2))
    else:
        raise Exception("The distance '%s' is not supported." % distance)
