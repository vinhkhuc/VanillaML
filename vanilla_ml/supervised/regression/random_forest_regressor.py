"""
Random forest classifier
"""
import numpy as np

from vanilla_ml.base.random_forest import RandomForestBase
from vanilla_ml.supervised.regression.abstract_regressor import AbstractRegressor


class RandomForestRegressor(RandomForestBase, AbstractRegressor):

    def __init__(self,
                 num_trees=10,
                 max_depth=1,
                 criterion='mse',
                 min_leaf_samples=1,
                 rand_samples_ratio=None,  # bagging is used if specified
                 rand_features_ratio=0.7,
                 rand_state=42,
                 verbose=False):
        super(RandomForestRegressor, self).__init__(num_trees,
                                                    max_depth,
                                                    criterion,
                                                    min_leaf_samples,
                                                    rand_samples_ratio,
                                                    rand_features_ratio,
                                                    rand_state,
                                                    verbose)

    def fit(self, X, y, sample_weights=None):
        super(RandomForestRegressor, self).fit(X, y, sample_weights)

    def predict(self, X):
        # Get prediction from each decision tree
        y_prob_trees = np.array([tree.predict(X) for tree in self.trees])

        # Return averaged prediction
        return y_prob_trees.mean(axis=0)

