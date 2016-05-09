"""
Utility to compute various distances
"""
import numpy as np


def compute_dist_matrix(X, cluster_centroids, distance):
    """ Compute the distance matrix where each element m[i][j] is
    the distance between X[i] and centroids[j].

    Args:
        X (ndarray): samples, shape N x P.
        cluster_centroids (ndarray): cluster centroids, shape K x P.
        distance (str): distance type.

    Returns:
        ndarray: distance matrix, shape N x K.

    """
    N, K = X.shape[0], cluster_centroids.shape[0]
    dist_matrix = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            dist_matrix[i][k] = dist(X[i], cluster_centroids[k], distance=distance)
    return dist_matrix


def dist(x, y, distance):
    """ Get distance between two data points.

    Args:
        x (ndarray): data point, shape P
        y (ndarray): data point, shape P
        distance (str): distance type

    Returns:
        float: distance between the given two data points
    """
    if distance == 'l2':
        return np.sqrt(np.sum(np.square(x - y)))
    else:
        raise Exception("The distance '%s' is not supported." % distance)
