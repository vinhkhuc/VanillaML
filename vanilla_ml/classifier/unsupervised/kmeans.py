"""
KMeans clustering
"""
import numpy as np
from vanilla_ml.classifier.unsupervised.abstract_clustering import AbstractClustering
from vanilla_ml.util import distances


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
            # NOTE: In order speed up the calculation, we can convert y to one-hot matrix.
            # Then centroids can be computed as centroids = y_one_hot.T * X
            cluster_sizes.fill(0)
            for x_i, y_i in zip(X, y):
                cluster_centroids[y_i] += x_i
                cluster_sizes[y_i] += 1
            cluster_centroids /= cluster_sizes[:, None]

            # Reassign samples to new clusters
            dist_matrix = distances.compute_dist_matrix(X, cluster_centroids, self.distance)
            next_y = dist_matrix.argmin(axis=1)

            # Check if clusters have changed
            if (y == next_y).all():
                break

            # Otherwise continue
            y = next_y

        # Save cluster centroids
        self.cluster_centroids = cluster_centroids

    def predict(self, X):
        dist_matrix = distances.compute_dist_matrix(X, self.cluster_centroids, self.distance)
        return dist_matrix.argmin(axis=1)
