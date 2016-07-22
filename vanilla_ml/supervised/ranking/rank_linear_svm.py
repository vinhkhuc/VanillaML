"""
RankLinearSVM using pairwise transformation and LinearSVM as the base classifier.

References:
    1) "Large Margin Rank Boundaries for Ordinal Regression",
        R. Herbrich, T. Graepel, K. Obermayer.
    2) https://gist.github.com/agramfort/2071994,
        by Fabian Pedregosa <fabian@fseoane.net>, Alexandre Gramfort <alexandre.gramfort@inria.fr>.
"""
import numpy as np

from vanilla_ml.supervised.classification.linear_svm import LinearSVM
from vanilla_ml.supervised.classification.logistic_regression import LogisticRegression
from vanilla_ml.supervised.ranking.abstract_ranker import AbstractRanker
from vanilla_ml.util import misc


class RankLinearSVM(AbstractRanker):

    def __init__(self, learning_rate=1.0, fit_bias=True, penalty_type=None, penalty_factor=1.0,
                 mini_batch_size=10, max_iterations=50, tol=1e-5, verbose=True, random_state=42):
        self.base_clf = LogisticRegression(learning_rate, fit_bias, penalty_type, penalty_factor,
                                           mini_batch_size, max_iterations, tol, verbose, random_state)

    def fit(self, X, y, sample_weights=None):
        assert sample_weights is None, "Sample weights are not supported."
        X_diff, y_diff = misc.pairwise_transform(X, y)
        y_diff = misc.sign_to_label(y_diff)
        self.base_clf.fit(X_diff, y_diff)

    def rank_score(self, X):
        if self.base_clf.fit_bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        return np.dot(X, self.base_clf.w)
