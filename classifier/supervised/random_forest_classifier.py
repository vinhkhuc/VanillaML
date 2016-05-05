"""
Random forest classifier
"""
import numpy as np

from classifier.supervised.abstract_classifier import AbstractClassifier
from classifier.supervised.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier(AbstractClassifier):

    def __init__(self,
                 num_trees=10,
                 max_depth=1,
                 criterion='entropy',
                 min_leaf_samples=1,
                 rand_samples_ratio=None,  # bagging is used if specified
                 rand_features_ratio=0.7,
                 rand_state=42,
                 verbose=False):
        super(RandomForestClassifier, self).__init__()

        assert min_leaf_samples > 0, "Minimum number of samples in leaf nodes must be positive."
        assert 0 < rand_features_ratio <= 1, "Ratio of random features must be in (0, 1]."

        tree = DecisionTreeClassifier(max_depth=max_depth,
                                      criterion=criterion,
                                      min_leaf_samples=min_leaf_samples,
                                      rand_features_ratio=rand_features_ratio,
                                      rand_state=rand_state,
                                      verbose=verbose)
        self.trees = [tree] * num_trees
        self.rand_samples_ratio = rand_samples_ratio
        np.random.seed(rand_state)

    def fit(self, X, y):
        super(RandomForestClassifier, self).fit(X, y)

        for tree in self.trees:
            # Select a random subset of samples if bagging is used
            if self.rand_samples_ratio is not None:
                total_samples = X.shape[0]
                num_rand_samples = int(total_samples * self.rand_samples_ratio)
                rand_samples = np.random.randint(0, num_rand_samples)
                rand_samples.sort()
                X = X[rand_samples, :]
                y = y[rand_samples]

            # Fit
            tree.fit(X, y)

    def predict_proba(self, X):
        # Get prediction probability for each decision tree
        y_prob_trees = np.array([tree.predict_proba(X) for tree in self.trees])

        print("y_prob_trees = %s" % y_prob_trees)
        print("y_prob_trees.mean(axis=0)\n%s" % y_prob_trees.mean(axis=0))

        # Return probability average
        return y_prob_trees.mean(axis=0)
