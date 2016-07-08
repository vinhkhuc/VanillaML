"""
Random forest works by averaging results of multiples small decision trees.
"""
import numpy as np

from vanilla_ml.supervised.classification.decision_tree_classifier import DecisionTreeClassifier
from vanilla_ml.supervised.regression.decision_tree_regressor import DecisionTreeRegressor


class RandomForestBase(object):
    """
    Base class for random forest.
    """
    def __init__(self,
                 num_trees=10,
                 max_depth=1,
                 criterion=None,
                 min_leaf_samples=1,
                 rand_samples_ratio=None,  # bagging is used if specified
                 rand_features_ratio=0.7,
                 rand_state=42,
                 verbose=False):
        assert min_leaf_samples > 0, "Minimum number of samples in leaf nodes must be positive."
        assert 0 < rand_features_ratio <= 1, "Ratio of random features must be in (0, 1]."

        self.rand_samples_ratio = rand_samples_ratio
        np.random.seed(rand_state)

        is_classifier = criterion == 'gini' or criterion == 'entropy'

        self.trees = []
        for _ in range(num_trees):
            if is_classifier:
                tree = DecisionTreeClassifier(max_depth=max_depth,
                                              criterion=criterion,
                                              min_leaf_samples=min_leaf_samples,
                                              rand_features_ratio=rand_features_ratio,
                                              rand_state=rand_state,
                                              verbose=verbose)
            else:
                tree = DecisionTreeRegressor(max_depth=max_depth,
                                             criterion=criterion,
                                             min_leaf_samples=min_leaf_samples,
                                             rand_features_ratio=rand_features_ratio,
                                             rand_state=rand_state,
                                             verbose=verbose)
            self.trees.append(tree)

    def fit(self, X, y, sample_weights=None):
        for tree in self.trees:
            # Select a random subset of samples if bagging is used.
            if self.rand_samples_ratio is not None:
                total_samples = X.shape[0]
                num_rand_samples = int(total_samples * self.rand_samples_ratio)
                rand_samples = np.random.randint(0, num_rand_samples)
                rand_samples.sort()
                X = X[rand_samples, :]
                y = y[rand_samples]
            # Fit
            tree.fit(X, y)
