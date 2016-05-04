import copy
import math

import numpy as np

from classifier.supervised.abstract_classifier import AbstractClassifier

SMALL_EPS = 1e-10

"""
Classic AdaBoost which is a binary classifier.
Ref: Algorithm 1.1 (p.5) in the book "Boosting: Foundations and Algorithms" (by Robert E. Schapire and Yoav Freund).
"""
class AdaBoostClassifier(AbstractClassifier):

    def __init__(self, base_clf, num_rounds, verbose=False):
        super(AdaBoostClassifier, self).__init__()

        self.base_clf = base_clf
        self.num_rounds = num_rounds
        self.verbose = verbose
        self.alphas = []
        self.hs = []  # hypotheses

    def fit(self, X, y):
        super(AdaBoostClassifier, self).fit(X, y)

        N = len(X)
        D = np.ones(N) / N

        for i in range(self.num_rounds):
            # Create a new base classifier with the same settings
            hs_i = copy.copy(self.base_clf)
            hs_i.fit(X, y, sample_weight=D)

            # Fit and get weighted errors
            pred_y = hs_i.predict(X)
            errors = (pred_y != y).astype(int)
            eps = np.sum(errors * D)  # weighted error

            # Early stopping
            if eps > 0.5:
                if self.verbose:
                    print("Weighted error is larger than the error from random guess (%g > 0.5). "
                          "Will stop boosting." % eps)
                break

            # Compute alpha (using SMALL_EPS to avoid computing log(0))
            alphas_i = 0.5 * math.log((1 - eps + SMALL_EPS) / (eps + SMALL_EPS))

            # Update instance weights
            yh_i = _sign_pred(y) * _sign_pred(pred_y)
            exp_alphas = np.exp(-alphas_i * yh_i)
            D *= exp_alphas
            D /= sum(D)

            # Save boosted learner
            self.alphas.append(alphas_i)
            self.hs.append(hs_i)

            if self.verbose:
                print("\n%d)" % (i + 1))
                print("eps = %s" % eps)
                print("errors = %s" % errors)
                print("total errors = %d" % len(np.where(errors > 0)[0]))
                print("alphas_i = %s" % alphas_i)
                print("exp_alphas = %s" % exp_alphas)
                print("D = %s" % D)
                pred_train_y = self.predict(X)
                train_errors = np.mean(pred_train_y != y)
                print("train_errors = %g" % train_errors)

    def predict(self, X):
        hxs = np.array([_sign_pred(hs_i.predict(X)) for hs_i in self.hs])
        tmp = np.dot(hxs.T, self.alphas)
        sign_pred_y = np.sign(tmp)
        return _unsign_pred(sign_pred_y)

    def predict_proba(self, X):
        raise Exception("AdaBoostClassifier")

def _sign_pred(y):
    """
    Maps {0, 1} to {-1, 1}.
    """
    return 2 * y - 1


def _unsign_pred(y):
    """
    Maps {-1, 1} to {0, 1}.
    """
    return (y + 1) / 2
