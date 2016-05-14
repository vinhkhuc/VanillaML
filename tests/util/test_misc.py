import unittest

import numpy as np

from vanilla_ml.util import misc


class TestMisc(unittest.TestCase):

    def test_logsumexp(self):
        X = np.array([[1.0, 3.0, 5.0]])
        log_sum_exp_X = misc.log_sum_exp(X)
        print("log_sum_exp_X = %s" % log_sum_exp_X)

        exp_X = np.exp(X)
        sum_exp_X = exp_X.sum(axis=1)
        log_sum_exp_X_naive = np.log(sum_exp_X)
        print("log_sum_exp_X_naive = %s" % log_sum_exp_X_naive)

        self.assertTrue(np.allclose(log_sum_exp_X, log_sum_exp_X_naive))

    def test_softmax(self):
        X = np.array([[1.0, 3.0, 5.0]])
        softmax_X = misc.softmax(X)
        print("softmax_X = %s" % softmax_X)

        exp_X = np.exp(X)
        sum_exp_X = exp_X.sum(axis=1)
        softmax_X_naive = exp_X / sum_exp_X[:, None]
        print("softmax_X_naive = %s" % softmax_X_naive)

        self.assertTrue(np.allclose(softmax_X, softmax_X_naive))
