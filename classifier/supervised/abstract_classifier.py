from abc import ABCMeta, abstractmethod

import numpy as np


class AbstractClassifier(object):
    """
    Abstract classifier
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self._classes = None
        pass

    @abstractmethod
    def fit(self, X, y):
        """ Fit the model using the given training data set with n data points and p features.

        Args:
            X (ndarray): training data set, shape N x P.
            y (ndarray): training labels, shape N x 1.

        """
        assert len(X) == len(y), "Length mismatches: len(X) = %d, len(y) = %d" % (len(X), len(y))
        assert np.all(y >= 0), "y must be non-negative"
        self._classes = np.unique(y)

    @abstractmethod
    def predict_proba(self, X):
        """ Predict outcome's probabilities for the testing set.

        Args:
            X (ndarray): test set, shape M x P.

        Returns:
            ndarray: outcome's probabilities: numpy array n x c where c is the number of classes

        """
        pass

    def predict(self, X):
        """ Predict outcomes for the testing set.

        Args:
            X (ndarray): test set, shape M x P.

        Returns:
            ndarray: predicted outcomes, shape N x 1.

        """
        y_pred = self.predict_proba(X)
        return y_pred.argmax(axis=1)
