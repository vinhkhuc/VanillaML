from abc import ABCMeta, abstractmethod


class AbstractScaler(object):
    """
    Abstract scaler
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
