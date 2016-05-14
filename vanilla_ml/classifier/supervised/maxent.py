"""
Maximum Entropy
"""
import numpy as np
from vanilla_ml.classifier.supervised.abstract_classifier import AbstractClassifier
from vanilla_ml.util import misc


# TODO: Give warning if X is detected as non-scaled
class MaxEnt(AbstractClassifier):
    """
    Maximum Entropy trained using SGD.
    Based on the gradient formula 8.38 in Kevin Murphy's book.
    """
    ALLOWED_PENALTIES = {'l1', 'l2'}

    def __init__(self, penalty_type=None, learning_rate=1.0, penalty_factor=1.0,
                 mini_batch_size=10, max_iterations=50, random_state=42):

        assert learning_rate > 0, "Learning rate must be positive."

        if penalty_type is not None:
            assert penalty_type in MaxEnt.ALLOWED_PENALTIES, \
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

    def fit(self, X, y, sample_weights=None):
        assert sample_weights is None, "Sample weights are not supported!"
        assert len(X) == len(y), "Length mismatches: len(X) = %d, len(y) = %d" % (len(X), len(y))

        np.random.seed(self.random_state)

        y = y.astype(int)
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Convert y to one-hot matrix
        one_hot_y = misc.one_hot(y, n_classes)

        n_samples, n_features = X.shape
        self.w = np.zeros((n_classes, n_features))
        # self.w = np.random.randn(n_classes, n_features)

        # SGD
        indices = np.arange(n_samples)
        for it in range(self.max_iterations):
            if (it + 1) % 10 == 0:
                print("Iteration %d ..." % (it + 1))

            # Check for convergence
            pred_y = self.predict(X)
            print("Iteration %d\n\ty = %s\n\tpred_y = %s" % (it + 1, y, pred_y))
            if (pred_y == y).all():
                break

            # Update w
            mini_batch = np.random.choice(indices, size=self.mini_batch_size, replace=False)
            X_batch, one_hot_y_batch = X[mini_batch], one_hot_y[mini_batch]
            grad = self._grad(X_batch, one_hot_y_batch)
            self.w -= (1. / self.mini_batch_size) * self.lr * grad

            if it == self.max_iterations - 1:
                print("Maximum iterations has reached.")

    def _grad(self, X, one_hot_y):
        pred_proba_y = self.predict_proba(X)  # shape = N x K
        print("predict_proba_y = %s" % pred_proba_y)

        mu = pred_proba_y - one_hot_y
        grad = np.dot(mu.T, X)  # grad's shape = K x P (same shape as w's)
        if self.penalty_type is not None:
            grad += misc.get_penalty(self.w, self.penalty_factor, self.penalty_type)

        return grad

    def predict_proba(self, X):
        Xw = np.dot(X, self.w.T)
        return misc.softmax(Xw)  # shape N x K
