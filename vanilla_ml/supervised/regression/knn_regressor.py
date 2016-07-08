"""
KNN Regressor
"""
import numpy as np

from vanilla_ml.base.knn import KNNBase
from vanilla_ml.supervised.regression.abstract_regressor import AbstractRegressor


class KNNRegressor(KNNBase, AbstractRegressor):

    def __init__(self, k, distance='l2'):
        KNNBase.__init__(self, k, distance)

    def fit(self, X, y, sample_weights=None):
        assert sample_weights is None, "Sample weights are not supported!"
        KNNBase.fit(self, X, y)

    def predict(self, test_X):
        nearest_indices = self._get_nearest_indices(test_X)
        pred_y = [np.mean(self.y[nearest_index_list])
                  for nearest_index_list in nearest_indices]
        return np.array(pred_y)
