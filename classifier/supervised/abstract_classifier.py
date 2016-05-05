from abc import ABCMeta, abstractmethod


class AbstractClassifier(object):
    """
    Abstract classifier
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X, y, sample_weights=None):
        """ Fit the model using the given training data set with n data points and p features.

        Args:
            X (ndarray): training data set, shape N x P.
            y (ndarray): training labels, shape N x 1.
            sample_weights (Optional[ndarray]): sample weights, shape N x 1.
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """ Predict outcome's probabilities for the testing set.

        Args:
            X (ndarray): test set, shape M x P.

        Returns:
            ndarray: outcome's probabilities, shape N x C where C is the number of classes.

        """
        pass

    def predict(self, X):
        """ Predict outcomes for the test set.

        Args:
            X (ndarray): test set, shape M x P.

        Returns:
            ndarray: predicted outcomes, shape N x 1.

        """
        y_pred = self.predict_proba(X)
        return y_pred.argmax(axis=1)
