import numpy as np

from vanilla_ml.base.decision_tree import DecisionTreeBase
from vanilla_ml.regression.abstract_regressor import AbstractRegressor


class DecisionTreeBaseRegressor(AbstractRegressor, DecisionTreeBase):

    def __init__(self,
                 max_depth=3,
                 criterion='mse',
                 min_leaf_samples=1,
                 rand_features_ratio=None,
                 rand_state=42,
                 verbose=False):

        assert criterion == 'mse', "The criterion '%s' is not supported by DecisionTreeRegressor." % criterion

        super(DecisionTreeBaseRegressor, self).__init__(max_depth, criterion, min_leaf_samples,
                                                        rand_features_ratio, rand_state, verbose)

    def predict(self, X):
        pass

    def fit(self, X, y):
        pass

    def _cost(self, y, w, criterion):
        """ Cost function

        Args:
            y (ndarray): sample classes N x 1
            w (ndarray): sample weights N x 1
            criterion (str): split criterion

        Returns:
            float: cost corresponding to the given criterion.
        """
        y_prob = self._get_weighted_predict_proba(y, w)
        if criterion == 'entropy':
            log2_y_prob = np.log2(y_prob)
            log2_y_prob[log2_y_prob == -np.inf] = 0  # replace -infs by zeros since they'll be eliminated anyway.
            return -np.sum(y_prob * log2_y_prob)
        elif criterion == 'gini':
            return np.sum(y_prob * (1 - y_prob))
        else:
            raise Exception("Criterion must be either entropy or gini.")
