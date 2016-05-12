"""
Gradient Boosted Regressor works by fitting multiple regressors where each regressor fits the residuals
between the previous targets and regressors.
"""
import copy

import numpy as np
from vanilla_ml.regression.abstract_regressor import AbstractRegressor


class GradientBoostedRegressor(AbstractRegressor):
    """
    This is still a simple boosting approach based on the Algorithm 8.2 (p. 322)
    from the book "Introduction to Statistical Learning with Applications in R".

    NOTE: alpha is actually lambda in the book.

    """
    def __init__(self, base_regr, num_rounds=10, alpha=1.0):
        """ Constructor

        Args:
            base_regr (AbstractRegressor): base regressor
            num_rounds (int): number of boosting rounds
            alpha (float): boosting rate

        """
        self.regrs = [copy.copy(base_regr) for _ in range(num_rounds)]
        self.num_rounds = num_rounds
        self.alpha = alpha

    def fit(self, X, y):
        """ Fit the boosted regressor.

        Args:
            X (ndarray): training examples, shape N x P.
            y (ndarray): training targets, shape N x 1

        """
        # TODO: For tree regressor, support the parameter maximum splits 'd' (as shown in the Algorithm 8.2).
        r = y  # residuals
        for i in range(self.num_rounds):
            base_regr = self.regrs[i]
            base_regr.fit(X, r)
            pred_y = base_regr.predict(X)
            r -= self.alpha * pred_y

    def predict(self, X):
        # Get prediction from each boosted regressor
        pred_ys = np.array([base_regr.predict(X) for base_regr in self.regrs])

        # Return additive results
        return self.alpha * pred_ys.sum(axis=0)
