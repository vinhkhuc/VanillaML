from abc import ABCMeta, abstractmethod


class AbstractRegressor(object):
    """
    Abstract regressor
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X, y):
        """ Fit the model using the given training data set with n data points and p features.

        Args:
            X (ndarray): training examples, shape N x P.
            y (ndarray): training labels, shape N x 1.

        """
        pass

    @abstractmethod
    def predict(self, X):
        """ Predict outcomes for the test set.

        Args:
            X (ndarray): test examples, shape M x P.

        Returns:
            ndarray: predicted outcomes, shape N x 1.

        """
        pass
