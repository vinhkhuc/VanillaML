"""
Decision tree works by recursively splitting the input space into regions
and create local model for each region.
"""
import numpy as np

from vanilla_ml.base.decision_tree import DecisionTreeBase
from vanilla_ml.classifier.supervised.abstract_classifier import AbstractClassifier

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
        DecisionTreeBase.fit(self, X, y, sample_weights)

    def predict_proba(self, X):
        return DecisionTreeBase.predict(self, X)
