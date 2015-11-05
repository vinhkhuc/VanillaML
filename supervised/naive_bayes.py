import math
import numpy as np
import scipy as sp
from collections import Counter
from abstract_classifier import AbstractClassifier

# Time complexity:
#   Training time   = O(N * P), where N - number of training samples, P - number of features
#   Prediction time =
class NaiveBayes(AbstractClassifier):

    def __init__(self):
        super(NaiveBayes, self).__init__()

        # A list of dictionaries mapping each feature to its per-class frequency
        self.feat_freq_by_class = None

        # A list of class prior
        self._class_prior = None

    def fit(self, tr_X, tr_y):
        """
        Fit model using the given training data set
        @param tr_X: numpy array of integers n x p (n - number of training samples, p - number of features)
        @param tr_y: numpy array n x 1
        @return: self NaiveBayes object
        """
        class_freq = Counter(tr_y)
        self._classes = np.array(sorted(class_freq.keys()))

        self._class_prior = np.array([class_freq[_class] for _class in self._classes], dtype=np.float)
        self._class_prior /= self._class_prior.sum()

        self.feat_freq_by_class = []
        for _class in self._classes:
            tr_X_by_class = tr_X[tr_y == _class]
            feat_freq = tr_X_by_class.sum(axis=0)  # sum by rows

            # Convert Numpy matrix to 1-D Numpy array
            feat_freq_dense = np.asarray(feat_freq).ravel()

            self.feat_freq_by_class.append(
                dict([(i, freq) for i, freq in enumerate(feat_freq_dense) if freq > 0]))

        return self

    def predict(self, te_X):
        """
        Predict outcomes for the testing set
        @param te_X: numpy array
        @return: predicted outcomes: numpy array n x 1
        """
        # log_probs = self._compute_log_probs(te_X)
        # return log_probs.argmax(axis=1)

        # TODO: Uncomment the above block
        # Here we just want to make sure that predict_proba is correctly implemented.
        probs = self.predict_proba(te_X)
        return probs.argmax(axis=1)

    def predict_proba(self, te_X):
        """
        Predict outcome's probabilities for the testing set
        @param te_X: numpy array
        @return: outcome's probabilities:
                 an Numpy array N x C where N is the number of samples, C is the number of classes
        """
        log_probs = self._compute_log_probs(te_X)

        # probs = np.empty_like(log_probs)
        # for i, log_prob in enumerate(log_probs):
        #     probs[i] = np.exp(log_prob - NaiveBayes._log_sum_exp(log_prob))

        # compute_prob = lambda log_prob: np.exp(log_prob - NaiveBayes._log_sum_exp(log_prob))
        # probs = np.array([compute_prob(log_prob) for log_prob in enumerate(log_probs)])

        probs = np.apply_along_axis(func1d=lambda log_prob: np.exp(log_prob - NaiveBayes._log_sum_exp(log_prob)),
                                    axis=1, arr=log_probs)

        return probs

    def _compute_log_probs(self, te_X):
        """
        Compute log probabilities for each sample in te_X
        @return: a Numpy array N x C
        """
        return np.array([[self._logPX_C(test_sample, class_idx)
                         for class_idx in range(len(self._classes))]
                         for i, test_sample in enumerate(te_X)])

    @staticmethod
    def _log_sum_exp(log_prob):
        """
        Computes log of sum of exps
        """
        max_val = log_prob.max()
        return np.log(np.sum(np.exp(log_prob - max_val))) + max_val  # log(sum(exp)) + max :)

    def _smoothed_feature_freq(self, feat_idx, class_idx, smoothing_factor=1.):
        """
        Computes count(w_i|C)
        We assume count(w_j|C) = 1 if the feature w_j doesn't occur in the training sample from the class C
        """
        return smoothing_factor + self.feat_freq_by_class[class_idx].get(feat_idx, 0)

    def _logPX_C(self, test_sample, class_idx):
        """
        Computes the conditional probability P(C|X) ~ P(X|C) * P(C)
                                                    ~ P(w_1|C) ... P(w_n|C) * P(C)
        """
        test_sample_cx = sp.sparse.coo_matrix(test_sample)
        feature_freqs = np.array([[feat_val, self._smoothed_feature_freq(feat_idx, class_idx)]
                                  for _, feat_idx, feat_val in zip(test_sample_cx.row,
                                                                   test_sample_cx.col,
                                                                   test_sample_cx.data)])
        feature_probs = feature_freqs[:, 1]
        feature_probs /= feature_probs.sum()

        log_prob_features = feature_freqs[:, 0] * np.log(feature_probs)
        sum_log_prob_features = log_prob_features.sum()

        return sum_log_prob_features + math.log(self._class_prior[class_idx])
