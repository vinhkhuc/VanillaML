"""
Classic perceptron, averaged perceptron
"""
import numpy as np
from vanilla_ml.supervised.classification.abstract_classifier import AbstractClassifier
from vanilla_ml.util import misc
from vanilla_ml.util.misc import label_to_sign, sign_to_label


# TODO: Check convergence (check maxent.py)
# TODO: Add AveragedPerceptron
class Perceptron(AbstractClassifier):
    """
    Classic perceptron (see the Algorithm 8.4 in Kevin Murphy's book).
    """
    def __init__(self, fit_bias=True, max_iterations=50):
        self.fit_bias = fit_bias
        self.max_iterations = max_iterations
        self._classes = None
        self.w = None

    def fit(self, X, y, sample_weights=None):
        assert sample_weights is None, "Sample weights are not supported!"
        assert len(X) == len(y), "Length mismatches: len(X) = %d, len(y) = %d" % (len(X), len(y))

        y = y.astype(int)
        assert np.all(y >= 0) and np.all(y <= 1), "y must contain either 0 or 1."

        if self.fit_bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        self._classes = np.unique(y)
        sign_y = label_to_sign(y)

        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        for it in range(self.max_iterations):
            i = it % n_samples
            pred_sign_yi = misc.sign(np.inner(w, X[i]))
            if pred_sign_yi != sign_y[i]:
                w += sign_y[i] * X[i]
            else:
                # Check convergence (in an inefficient way!)
                pred_sign_y = np.sign(np.dot(X, w))
                if (sign_y == pred_sign_y).all():
                    break

            if it == self.max_iterations - 1:
                print("Maximum iterations has reached.")

        self.w = w

    def predict_proba(self, X):
        raise Exception("Perception doesn't support probability prediction.")

    def predict(self, X):
        if self.fit_bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        pred_sign_y = np.sign(np.dot(X, self.w))
        return sign_to_label(pred_sign_y)
