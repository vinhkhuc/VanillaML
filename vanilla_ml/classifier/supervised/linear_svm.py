"""
Support Vector Machines
"""
import numpy as np
from vanilla_ml.classifier.supervised.abstract_classifier import AbstractClassifier
from vanilla_ml.util import misc
from vanilla_ml.util.misc import sign_prediction, unsign_prediction


class LinearSVM(AbstractClassifier):
    """
    Linear SVM trained using SGD
    """
    ALLOWED_PENALTIES = {'l1', 'l2'}

    def __init__(self, learning_rate=1.0, fit_bias=True, penalty_type=None, penalty_factor=1.0,
                 mini_batch_size=10, max_iterations=50, random_state=42):

        assert learning_rate > 0, "Learning rate must be positive."

        if penalty_type is not None:
            assert penalty_type in LinearSVM.ALLOWED_PENALTIES, \
                "Penalty '%s' is not supported!" % penalty_type
            assert penalty_factor > 0, "Penalty factor must be positive."

        self.lr = learning_rate
        self.fit_bias = fit_bias
        self.penalty_type = penalty_type
        self.penalty_factor = penalty_factor
        self.mini_batch_size = mini_batch_size
        self.max_iterations = max_iterations
        self.random_state = random_state
        self._classes = None
        self.w = None

    # TODO: Fit bias!!!
    # TODO: Check if pred_y doesn't change to stop the iterations
    # TODO: Give warning if X is detected as non-scaled
    def fit(self, X, y, sample_weights=None):
        assert sample_weights is None, "Sample weights are not supported!"
        assert len(X) == len(y), "Length mismatches: len(X) = %d, len(y) = %d" % (len(X), len(y))

        y = y.astype(int)
        assert np.all(y >= 0) and np.all(y <= 1), "y must contain either 0 or 1."

        if self.fit_bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        sign_y = sign_prediction(y)
        self._classes = np.unique(sign_y)

        np.random.seed(self.random_state)
        indices = np.arange(n_samples)

        # Stochastic Gradient descent
        for it in range(self.max_iterations):
            if (it + 1) % 10 == 0:
                print("Iteration %d ..." % (it + 1))

            # Check for convergence
            sign_pred_y = np.sign(np.dot(X, self.w))
            if (sign_pred_y == sign_y).all():
                break

            if (it + 1) % 10 == 0:
                print("sign_pred_y = %s" % sign_pred_y)

            # Update w
            mini_batch = np.random.choice(indices, size=self.mini_batch_size, replace=False)
            X_batch, sign_y_batch = X[mini_batch], sign_y[mini_batch]
            grad = self._grad(X_batch, sign_y_batch)
            self.w -= (1. / self.mini_batch_size) * self.lr * grad

            if it == self.max_iterations - 1:
                print("Maximum iterations has reached.")

    def _grad(self, X, sign_y):
        # Subgradient of hinge-loss
        # L = max(0, 1 - w * yX) = 1 - w * yX if w * yX <= 1 else 0
        # grad = -yX if w * yX <= 1 else 0
        yX = np.dot(X.T, sign_y)
        grad = -yX if np.inner(self.w, yX) <= 1 else 0
        if self.penalty_type is not None:
            grad += misc.get_penalty(self.w, self.penalty_factor, self.penalty_type)

        return grad

    def predict_proba(self, X):
        raise Exception("Linear SVM doesn't support predict_proba")

    def predict(self, X):
        if self.fit_bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        pred_sign_y = np.sign(np.dot(X, self.w))
        return unsign_prediction(pred_sign_y)
