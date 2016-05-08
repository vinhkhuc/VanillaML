"""
KMeans clustering
"""
import numpy as np
from vanilla_ml.classifier.unsupervised.abstract_clustering import AbstractClustering


class KMeans(AbstractClustering):

    ALLOWED_DISTANCES = {'l2'}
    ALLOWED_INIT = {'random'}

    def __init__(self, n_clusters=10, distance='l2', init='random', random_state=42):
        """ Constructor

        Args:
            n_clusters (int): number of clusters
            distance (str): distance that will be used.
            init (str): initialization method.

        """
        assert n_clusters > 0, "Number of clusters must be larger than zero."
        assert distance in KMeans.ALLOWED_DISTANCES, "The distance '%s' is not supported." % distance
        assert init in KMeans.ALLOWED_INIT, "The initialization '%s' is not supported." % init

        self.n_clusters = n_clusters
        self.cluster_centroids = None
        self.distance = distance
        self.init = init

        np.random.seed(random_state)

    def fit(self, X, sample_weights=None):
        assert sample_weights is None, "Setting sample weights is not currently supported."

        n_samples, n_features = X.shape
        n_clusters = self.n_clusters

        # Cluster centroids
        cluster_centroids = np.zeros((n_clusters, n_features), dtype=float)
        cluster_sizes = np.empty(n_clusters)

        # Assign each sample to a random cluster
        y = np.random.randint(0, n_clusters, size=n_samples)

        while True:
            # Calculate cluster centroids.
            # NOTE: In order speed up the calculation, centroid calculate we can
            # convert y to one-hot matrix. Then, centroids = y_one_hot.T * X
            cluster_sizes.fill(0)
            for x_i, y_i in zip(X, y):
                cluster_centroids[y_i] += x_i
                cluster_sizes[y_i] += 1
            cluster_centroids /= cluster_sizes[:, None]

            # Reassign samples to new clusters
            dist_matrix = _get_dist_matrix(X, cluster_centroids, self.distance)
            next_y = dist_matrix.argmin(axis=1)

            # Check if clusters have changed
            if (y == next_y).all():
                break

            # Otherwise continue
            y = next_y

        # Save cluster centroids
        self.cluster_centroids = cluster_centroids

    def predict(self, X):
        dist_matrix = _get_dist_matrix(X, self.cluster_centroids, self.distance)
        return dist_matrix.argmin(axis=1)


def _get_dist_matrix(X, cluster_centroids, distance):
    """ Calculate a distance matrix so that each matrix element m[i][j] is
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
            dist_matrix[i][k] = _get_dist(X[i], cluster_centroids[k], distance=distance)
    return dist_matrix


def _get_dist(x, y, distance):
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
