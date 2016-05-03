from abc import ABCMeta, abstractmethod

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
        """
        Fit the model using the given training data set with n data points and p features
        ::param:: X: numpy array n x p
        ::param:: y: numpy array n x 1
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Predict outcome's probabilities for the testing set
        ::param:: X: numpy array
        @return outcome's probabilities: numpy array n x c where c is the number of classes
        """
        pass

    def predict(self, X):
        """
        Predict outcomes for the testing set
        ::param:: te_X: numpy array
        @return predicted outcomes: numpy array n x 1
        """
        y_pred = self.predict_proba(X)
        return y_pred.argmax(axis=1)
