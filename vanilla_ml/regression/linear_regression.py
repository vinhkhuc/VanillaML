"""
Linear regression (with L1, L2 regularization)
"""
import numpy as np
from vanilla_ml.regression.abstract_regressor import AbstractRegressor


class LinearRegressor(AbstractRegressor):
    """
    Linear regression trained using SGD.
    Solvers are based on formulas in Kevin Murphy's book:
        + Analytical solver is based on the formula 7.16 (OLS - Ordinal Least Squares).
        + L2-regularization solver is based on the formula 7.33 (Ridge Regression).
        + TODO: L1
    """
    ALLOWED_PENALTIES = {'l1', 'l2'}
    ALLOWED_SOLVERS = {'analytical', 'sgd'}

    def __init__(self, solver='analytical', fit_bias=True, learning_rate=1.0, penalty_type=None,
                 penalty_factor=1.0, mini_batch_size=10, max_iterations=50, random_state=42):

        assert solver in LinearRegressor.ALLOWED_SOLVERS, "The solver '%s' is invalid." % solver

        assert learning_rate > 0, "Learning rate must be positive."

        if penalty_type is not None:
            assert penalty_type in LinearRegressor.ALLOWED_PENALTIES, \
                "Penalty '%s' is not supported!" % penalty_type
            assert penalty_factor > 0, "Penalty factor must be positive."

        self.solver = solver
        self.fit_bias = fit_bias

        # Parameters needed by SGD solver
        self.lr = learning_rate
        self.penalty_type = penalty_type
        self.penalty_factor = penalty_factor
        self.mini_batch_size = mini_batch_size
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.w = None

    def fit(self, X, y, sample_weights=None):
        assert sample_weights is None, "Sample weights are not supported!"
        assert len(X) == len(y), "Length mismatches: len(X) = %d, len(y) = %d" % (len(X), len(y))

        if self.fit_bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        if self.solver == 'analytical':
            self._solve_analytical(X, y, sample_weights)
        else:
            self._solve_sgd(X, y, sample_weights)

    # Analytical solver
    def _solve_analytical(self, X, y, sample_weights):
        assert sample_weights is None, "Sample weights are not supported!"
        if self.penalty_type is None:
            self.w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        else:
            raise Exception("Regularization is not supported yet.")

    # SGD solver
    def _solve_sgd(self, X, y, sample_weights=None):
        assert sample_weights is None, "Sample weights are not supported!"
        raise Exception("TODO")

    def _grad(self, X, y):
        raise Exception("TODO")

    def predict(self, X):
        if self.fit_bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        return np.dot(X, self.w)
