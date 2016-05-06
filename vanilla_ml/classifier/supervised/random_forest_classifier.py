"""
Random forest works by averaging results of multiples small decision trees.
"""
import numpy as np

from vanilla_ml.base.random_forest import RandomForestBase
from vanilla_ml.classifier.supervised.abstract_classifier import AbstractClassifier


class RandomForestClassifier(RandomForestBase, AbstractClassifier):

    def __init__(self,
                 num_trees=10,
                 max_depth=1,
                 criterion='entropy',
                 min_leaf_samples=1,
                 rand_samples_ratio=None,  # bagging is used if specified
                 rand_features_ratio=0.7,
                 rand_state=42,
                 verbose=False):
        super(RandomForestClassifier, self).__init__(num_trees,
                                                     max_depth,
                                                     criterion,
                                                     min_leaf_samples,
                                                     rand_samples_ratio,
                                                     rand_features_ratio,
                                                     rand_state,
                                                     verbose)

    def fit(self, X, y, sample_weights=None):
        super(RandomForestClassifier, self).fit(X, y, sample_weights)

    def predict_proba(self, X):
        # Get prediction probability for each decision tree
        y_prob_trees = np.array([tree.predict_proba(X) for tree in self.trees])

        # Return averaged probability
        return y_prob_trees.mean(axis=0)
