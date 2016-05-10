"""
Logistic Regression
"""
import numpy as np
from vanilla_ml.classifier.supervised.abstract_classifier import AbstractClassifier
from vanilla_ml.util import misc


# FIXME: All predicted labels are zeros !!!
class LogisticRegression(AbstractClassifier):
    """
    Logistic regression trained using SGD.
    Based on the gradient formula 8.5 in Kevin Murphy's book
    """
    ALLOWED_PENALTIES = {'l1', 'l2'}

    def __init__(self, learning_rate=1.0, penalty_type=None, penalty_factor=1.0, max_iterations=50):
        assert learning_rate > 0, "Learning rate must be positive."
        assert penalty_type is None or penalty_factor > 0, "Penalty factor must be positive."

        self.lr = learning_rate
        self.penalty_type = penalty_type
        self.penalty_factor = penalty_factor
        self.max_iterations = max_iterations
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

        # Gradient descent
        # TODO: Use SGD
        for it in range(self.max_iterations):
            # Check for convergence
            pred_y = self.predict(X)
            if (pred_y == y).all():
                break

            # Update w
            grad = self._grad(X, y)
            self.w -= self.lr * grad

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
