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
    def fit(self, tr_X, tr_y):
        """
        Fit the model using the given training data set with n data points and p features
        ::param:: tr_X: numpy array n x p
        ::param:: tr_y: numpy array n x 1
        """
        assert len(tr_X) == len(tr_y), "Length mismatches: len(tr_X) = %d, len(tr_y) = %d" % (len(tr_X), len(tr_y))
        pass

    @abstractmethod
    def predict(self, te_X):
        """
        Predict outcomes for the testing set
        ::param:: te_X: numpy array
        @return predicted outcomes: numpy array n x 1
        """
        pass
