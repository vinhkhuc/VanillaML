import unittest

import numpy as np

from vanilla_ml.util.metrics import f1


class TestMetric(unittest.TestCase):

    def test_precision(self):
        pred_y = np.array([0, 0, 1, 1])
        true_y = np.array([1, 1, 1, 0])

        precision = f1.precision(pred_y, true_y)
        self.assertTrue(precision == 0.5)

    def recall(self):
        pred_y = np.array([0, 0, 1, 1])
        true_y = np.array([1, 1, 1, 0])

        recall = f1.recall(pred_y, true_y)
        assert(recall == 1. / 3)

    def test_f1(self):
        pred_y = np.array([0, 0, 1, 1])
        true_y = np.array([1, 1, 1, 0])

        f1_score = f1.f1_score(pred_y, true_y)
        self.assertTrue(f1_score == 0.4)
