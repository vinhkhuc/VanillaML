from abc import ABCMeta, abstractmethod


class AbstractClustering(object):
    """
    Abstract clustering
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X, sample_weights=None):
        """ Fit the model using the given training set.

        Args:
            X (ndarray): training data set, shape N x P.
            sample_weights (Optional[ndarray]): sample weights, shape N x 1.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """ Predict outcomes for the test set.

        Args:
            X (ndarray): test set, shape M x P.

        Returns:
            ndarray: predicted outcomes, shape N x 1.

        """
        pass
