"""
Decision tree works by recursively splitting the input space into regions
and create local model for each region.
"""
import numpy as np

from base.decision_tree import DecisionTreeBase
from classifier.supervised.abstract_classifier import AbstractClassifier

np.seterr(divide='ignore')  # ignore the warning message caused by calling log(0)


# FIXME: sklearn's decision tree gave 76% accuracy while ours got 72% for the moon dataset:
# train_X, test_X, train_y, test_y = get_moons_train_test()
class DecisionTreeBaseClassifier(AbstractClassifier, DecisionTreeBase):

    def __init__(self,
                 max_depth=3,
                 criterion='entropy',
                 min_leaf_samples=1,
                 rand_features_ratio=None,
                 rand_state=42,
                 verbose=False):

        assert criterion == 'entropy' or criterion == 'gini', \
            "The criterion '%s' is not supported by DecisionTreeClassifier." % criterion

        super(DecisionTreeBaseClassifier, self).__init__(max_depth, criterion, min_leaf_samples,
                                                         rand_features_ratio, rand_state, verbose)

    def fit(self, X, y, sample_weights=None):
        DecisionTreeBase(self).fit(X, y, sample_weights)

    def predict_proba(self, X):
        return DecisionTreeBase(self).predict_proba(X)

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
