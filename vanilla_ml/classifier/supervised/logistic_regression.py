"""
Logistic Regression
"""
import numpy as np
from vanilla_ml.classifier.supervised.abstract_classifier import AbstractClassifier
from vanilla_ml.util import misc


# FIXME: L2 regularization doesn't work
class LogisticRegression(AbstractClassifier):
    """
    Logistic regression trained using SGD.
    Based on the gradient formula 8.5 in Kevin Murphy's book.
    """
    ALLOWED_PENALTIES = {'l1', 'l2'}

    def __init__(self, learning_rate=1.0, penalty_type=None, penalty_factor=1.0,
                 mini_batch_size=10, max_iterations=50, random_state=42):

        assert learning_rate > 0, "Learning rate must be positive."
        if penalty_type is not None:
            assert penalty_type in LogisticRegression.ALLOWED_PENALTIES, \
                "Penalty '%s' is not supported!" % penalty_type
            assert penalty_factor > 0, "Penalty factor must be positive."

        self.lr = learning_rate
        self.penalty_type = penalty_type
        self.penalty_factor = penalty_factor
        self.mini_batch_size = mini_batch_size
        self.max_iterations = max_iterations
        self.random_state = random_state
        self._classes = None
        self.w = None

    # TODO: Give warning if X is detected as non-scaled
    def fit(self, X, y, sample_weights=None):
        assert sample_weights is None, "Sample weights are not supported!"
        assert len(X) == len(y), "Length mismatches: len(X) = %d, len(y) = %d" % (len(X), len(y))

        y = y.astype(int)
        assert np.all(y >= 0) and np.all(y <= 1), "y must contain either 0 or 1."

        self._classes = np.unique(y)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        np.random.seed(self.random_state)
        indices = np.arange(n_samples)

        # Stochastic Gradient descent
        for it in range(self.max_iterations):
            if (it + 1) % 10 == 0:
                print("Iteration %d ..." % (it + 1))

            # Check for convergence
            pred_y = self.predict(X)
            if (pred_y == y).all():
                break

            # Update w
            mini_batch = np.random.choice(indices, size=self.mini_batch_size, replace=False)
            X_batch, y_batch = X[mini_batch], y[mini_batch]
            grad = self._grad(X_batch, y_batch)
            self.w -= (1. / self.mini_batch_size) * self.lr * grad

            if it == self.max_iterations - 1:
                print("Maximum iterations has reached.")

    def _grad(self, X, y):
        pred_proba_y = self.predict_proba(X)
        grad = np.dot(pred_proba_y - y, X)
        if self.penalty_type is not None:
            grad += misc.get_penalty(self.w, self.penalty_factor, self.penalty_type)

        return grad

    def predict_proba(self, X):
        return misc.sigmoid(np.dot(X, self.w))

    def predict(self, X):
        y_pred_proba = self.predict_proba(X)
        y_pred = np.zeros_like(y_pred_proba, dtype=np.int)
        y_pred[y_pred_proba > 0.5] = 1
        return y_pred
