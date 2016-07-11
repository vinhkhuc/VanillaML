# Ref: Jurasky's book: https://web.stanford.edu/~jurafsky/slp3/7.pdf
# See the pseudo code in the Figure 7.2
from __future__ import division

import numpy as np

from vanilla_ml.supervised.classification.abstract_classifier import AbstractClassifier
from vanilla_ml.util import misc


class NaiveBayesClassifier(AbstractClassifier):
    """
    Naive Bayes classifier (currently works with dense matrices only, i.e. Numpy arrays)
    """
    def __init__(self, alpha=1):
        """ Naive Bayes classifier.

        Args:
            alpha (float): smoothing factor

        """
        self._alpha = alpha
        self._classes = None
        self._log_prior = None
        self._log_likelihood = None
        self._V = None

    def fit(self, X, y, sample_weights=None):
        assert sample_weights is None, "Sample weights are not supported in NaiveBayesClassifier"

        C = len(np.unique(y))
        N_doc, V = X.shape
        self._log_prior = np.log([(sum(y == c) / N_doc) for c in range(C)])
        self._log_likelihood = np.zeros((V, C), np.float)
        for c in range(C):
            count_c = X[y == c].sum(axis=0)
            sum_count_c = sum(count_c)
            self._log_likelihood[:, c] = np.log((count_c + self._alpha) / (sum_count_c + self._alpha))
        self._classes = C
        self._V = V

    def predict_proba(self, X):
        N_doc = X.shape[0]
        confidence_vals = np.empty((N_doc, self._classes), np.float)
        for i in range(N_doc):
            confidence_vals[i] = np.copy(self._log_prior)
            for w in range(self._V):
                if X[i][w] != 0:
                    confidence_vals[i] += self._log_likelihood[w, :]

        # Return probabilities
        return misc.softmax(confidence_vals)

# import math
# import numpy as np
# import scipy as sp
# from collections import Counter
# from abstract_classifier import AbstractClassifier
#
# """
# This is the implementation of Multinomial NB see section 2.1 in the paper:
# "Tackling the Poor Assumptions of Naive Bayes Text Classifiers"
# (http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)
#
# * Accuracy:
#     + digits: 85.33%
#         ===> 70.44% with Complement Multinomial NB
#
#     + 20newsgroups (word frequency): 79.50%
#         ==> 82.69% after feature selection using LinearSVC(C=1., penalty="l1", dual=False)
#
#         ==> 87.78% with Complement Multinomial NB ???
#
# * For reference, using sklearn's MultinomialNB (much faster)
#     + digits: 88.89%
#     + 20newsgroups (word frequency): 83.01% (
#         ==> 86.19% after feature selection using LinearSVC(C=1., penalty="l1", dual=False)
#
# * Time complexity:
#    Training time   = O(N * P), where N - number of training samples, P - number of features
#    Prediction time =
# """
#
#
# class NaiveBayesClassifier(AbstractClassifier):
#
#     def __init__(self):
#         self.feat_freq_by_class = None
#         self._class_prior = None
#         self._classes = None
#
#     def fit(self, X, y, sample_weights=None):
#         assert sample_weights is None, "Sample weights are not supported in NaiveBayesClassifier"
#
#         class_freq = Counter(y)
#         self._classes = np.array(sorted(class_freq.keys()))
#
#         self._class_prior = np.array([class_freq[_class] for _class in self._classes], dtype=np.float)
#         self._class_prior /= self._class_prior.sum()
#
#         self.feat_freq_by_class = []
#         for _class in self._classes:
#             tr_X_by_class = X[y == _class]
#             feat_freq = tr_X_by_class.sum(axis=0)  # sum by rows
#
#             # Convert Numpy matrix to 1-D Numpy array
#             feat_freq_dense = np.asarray(feat_freq).ravel()
#
#             self.feat_freq_by_class.append(
#                 dict([(i, freq) for i, freq in enumerate(feat_freq_dense) if freq > 0]))
#
#         return self
#
#     def predict_proba(self, X):
#         """
#         Predict outcome's probabilities for the test set
#         @param X: numpy array
#         @return: outcome's probabilities:
#                  an Numpy array N x C where N is the number of samples, C is the number of classes
#         """
#         log_probs = self._compute_log_probs(X)
#
#         # probs = np.empty_like(log_probs)
#         # for i, log_prob in enumerate(log_probs):
#         #     probs[i] = np.exp(log_prob - NaiveBayes._log_sum_exp(log_prob))
#
#         # compute_prob = lambda log_prob: np.exp(log_prob - NaiveBayes._log_sum_exp(log_prob))
#         # probs = np.array([compute_prob(log_prob) for log_prob in enumerate(log_probs)])
#
#         probs = np.apply_along_axis(func1d=lambda log_prob: np.exp(log_prob - _log_sum_exp(log_prob)),
#                                     axis=1, arr=log_probs)
#
#         return probs
#
#     def _compute_log_probs(self, te_X):
#         """
#         Compute log probabilities for each sample in te_X
#         @return: a Numpy array N x C
#         """
#         return np.array([[self._logPX_C(test_sample, class_idx)
#                          for class_idx in range(len(self._classes))]
#                          for i, test_sample in enumerate(te_X)])
#
#         # return np.array([[self._logPX_C_complement(test_sample, class_idx)
#         #                  for class_idx in range(len(self._classes))]
#         #                  for i, test_sample in enumerate(te_X)])
#
#     def _smoothed_feature_freq(self, feat_idx, class_idx, smoothing_factor=1.):
#         """
#         Computes count(w_i|C)
#         """
#         return smoothing_factor + self.feat_freq_by_class[class_idx].get(feat_idx, 0)
#
#     def _logPX_C(self, test_sample, class_idx):
#         """
#         Computes the conditional probability P(C|X) ~ P(X|C) * P(C)
#                                                     ~ P(w_1|C) ... P(w_n|C) * P(C)
#
#         To see how P(w_1|C) is estimated, see section 3.4.3 p. 79 from Kevin Murphy's book
#         """
#         test_sample_cx = sp.sparse.coo_matrix(test_sample)
#         feature_freqs = np.array([[feat_val, self._smoothed_feature_freq(feat_idx, class_idx)]
#                                   for _, feat_idx, feat_val in zip(test_sample_cx.row,
#                                                                    test_sample_cx.col,
#                                                                    test_sample_cx.data)])
#         feature_probs = feature_freqs[:, 1]
#         feature_probs /= feature_probs.sum()
#
#         feature_vals = feature_freqs[:, 0]
#         log_prob_features = feature_vals * np.log(feature_probs)
#         sum_log_prob_features = log_prob_features.sum()
#
#         # log_prob_features = np.log(feature_probs)
#         # sum_log_prob_features = log_prob_features.sum()
#
#         return sum_log_prob_features + math.log(self._class_prior[class_idx])
#
#         # TODO: Model P(x_i_j | C) as a multinomial distribution. Will it give better accuracy?
#         # No, since the factor will be canceled out when we compute posterior probability
#
#         # # See Murphy's ebook p.88 for Multinomial Bayesian
#         # log_multinomial_factor = np.log((1 + np.arange(feature_vals.sum())).sum()) - np.sum(np.log(feature_vals))
#         #
#         # return log_multinomial_factor + sum_log_prob_features + math.log(self._class_prior[class_idx])
#
#     def _smoothed_feature_freq_complement(self, feat_idx, class_idx, smoothing_factor=1.):
#         """
#         Computes count(w_i|C~)
#         """
#         feat_freq_complement = sum([self.feat_freq_by_class[i].get(feat_idx, 0)
#                                     for i in range(len(self._classes)) if i != class_idx])
#         return smoothing_factor + feat_freq_complement
#
#     def _logPX_C_complement(self, test_sample, class_idx):
#         """
#         Computes the complement conditional probability P(C~|X)
#         """
#         test_sample_cx = sp.sparse.coo_matrix(test_sample)
#         feature_freqs = np.array([[feat_val, self._smoothed_feature_freq_complement(feat_idx, class_idx)]
#                                   for _, feat_idx, feat_val in zip(test_sample_cx.row,
#                                                                    test_sample_cx.col,
#                                                                    test_sample_cx.data)])
#         feature_probs = feature_freqs[:, 1]
#         feature_probs /= feature_probs.sum()
#
#         feature_vals = feature_freqs[:, 0]
#         log_prob_features = feature_vals * np.log(feature_probs)
#         sum_log_prob_features = log_prob_features.sum()
#
#         return -sum_log_prob_features + math.log(self._class_prior[class_idx])
#
#
# def _log_sum_exp(log_prob):
#     """
#     Computes log of sum of exps
#     """
#     max_val = log_prob.max()
#     return np.log(np.sum(np.exp(log_prob - max_val))) + max_val  # log(sum(exp)) + max :)
