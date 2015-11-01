import math
import numpy as np
import scipy as sp
from collections import Counter
from abstract_classifier import AbstractClassifier

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
        ::param:: tr_X: numpy array of integers n x p (n - number of training samples, p - number of features)
        ::param:: tr_y: numpy array n x 1
        @return self
        """
        class_freq = Counter(tr_y)
        self._classes = np.array(sorted(class_freq.keys()))

        self._class_prior = np.array([class_freq[_class] for _class in self._classes], dtype=np.float)
        self._class_prior /= self._class_prior.sum()

        self.feat_freq_by_class = []
        for i, _class in enumerate(self._classes):
            tr_X_by_class = tr_X[tr_y == _class]
            feat_freq = tr_X_by_class.sum(axis=0)  # sum by rows
            feat_freq_dense = feat_freq.A1 # convert to 1-D numpy array
            self.feat_freq_by_class.append(
                dict([(i, freq) for i, freq in enumerate(feat_freq_dense) if freq > 0]))

        return self

    def predict(self, te_X):
        """
        Predict outcomes for the testing set
        ::param:: te_X: numpy array
        @return predicted outcomes: numpy array n x 1
        """
        predictions = []
        for i, test_sample in enumerate(te_X):
            log_probs = np.array([self._logPX_C(test_sample, class_idx)
                                  for class_idx in range(len(self._classes))])
            predictions.append(log_probs.argmax())
        return np.array(predictions)

    def _log_prob_feature(self, feat_idx, class_idx):
        """
        Computes log P(w_i|C)
        We assume P(w_j|C) = 1e-8 if the feature w_j doesn't occur in the training sample from the class C
        """
        return math.log(self.feat_freq_by_class[class_idx].get(feat_idx, 1e-8))

    def _logPX_C(self, test_sample, class_idx):
        """
        Computes the conditional probability P(C|X) ~ P(X|C) * P(C) ~ P(w1|C) ... P(wn|C) * P(C)
        """
        test_sample_cx = sp.sparse.coo_matrix(test_sample)
        sum_log_prob_feat = sum([self._log_prob_feature(feat_idx, class_idx)
                                 for _, feat_idx, feat_val in zip(test_sample_cx.row,
                                                                  test_sample_cx.col,
                                                                  test_sample_cx.data)
                                 if feat_val > 0])

        return sum_log_prob_feat + math.log(self._class_prior[class_idx])
