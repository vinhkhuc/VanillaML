"""
KNN
"""
from vanilla_ml.util import distances


class KNNBase(object):

    ALLOWED_DISTANCES = {'l2'}

    def __init__(self, k, distance='l2'):
        assert isinstance(k, int) and k > 0, "K must be a positive integer."
        assert distance in KNNBase.ALLOWED_DISTANCES, "The distance '%s' is not supported." % distance
        self.k = k
        self.distance = distance
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def _get_nearest_indices(self, test_X):
        dist_matrix = distances.compute_dist_matrix(test_X, self.X, self.distance)
        sorted_idx = dist_matrix.argsort(axis=1)
        return sorted_idx[:, :self.k]
