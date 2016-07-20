from abc import ABCMeta, abstractmethod


class AbstractRanker(object):
    """
    Abstract ranker
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X, y, sample_weights=None):
        """ Fit the model using the given training data set with n data points and p features.

        Args:
            X (ndarray): training data set, shape N x P.
            y (ndarray): training ranks, shape N x 1.
            sample_weights (Optional[ndarray]): sample weights, shape N x 1.
        """
        pass

    @abstractmethod
    def rank_score(self, X):
        """ Compute ranking scores for the test set.

        Args:
            X (ndarray): test set, shape M x P.

        Returns:
            ndarray: ranking scores, shape N.

        """
        pass

    def rank(self, X):
        """ Rank elements from the test set. The elements are sorted in descending
        order of ranking scores.

        Args:
            X (ndarray): test set, shape M x P.

        Returns:
            ndarray: ranked element's indices, shape N.

        """
        scores = self.rank_score(X).ravel()
        return scores.argsort()[::-1]
