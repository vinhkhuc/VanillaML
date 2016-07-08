"""
KNN classifier
"""
import numpy as np
from vanilla_ml.base.knn import KNNBase
from vanilla_ml.supervised.classification.abstract_classifier import AbstractClassifier


class KNNClassifier(AbstractClassifier, KNNBase):

    def __init__(self, k, distance='l2'):
        KNNBase.__init__(self, k, distance)
        self._classes = None

    def fit(self, X, y, sample_weights=None):
        assert sample_weights is None, "Sample weights are not supported!"
        self._classes = np.unique(y)
        KNNBase.fit(self, X, y)

    def predict_proba(self, test_X):
        N = test_X.shape[0]
        n_classes = len(self._classes)
        nearest_indices = self._get_nearest_indices(test_X)

        pred_y = np.zeros((N, n_classes))
        for i in range(N):
            nearest_labels = self.y[nearest_indices[i]]
            y_prob = np.bincount(nearest_labels) / float(len(nearest_labels))
            diff = n_classes - len(y_prob)  # fill zeros for classes that are not included in y
            y_prob = np.pad(y_prob, (0, diff), mode='constant', constant_values=0)
            pred_y[i] = y_prob

        return pred_y
