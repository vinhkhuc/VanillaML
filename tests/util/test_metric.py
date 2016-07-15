import unittest

import numpy as np

from vanilla_ml.util.metrics import f1, ranking


class TestMetric(unittest.TestCase):

    def test_precision(self):
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1])

        precision = f1.precision(y_true, y_pred)
        self.assertTrue(precision == 0.5)

    def recall(self):
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1])

        recall = f1.recall(y_true, y_pred)
        assert(recall == 1. / 3)

    def test_f1(self):
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1])

        f1_score = f1.f1_score(y_true, y_pred)
        self.assertTrue(f1_score == 0.4)

    def test_dcg(self):
        y_true = np.array([3, 3, 2, 2, 1, 0])
        y_pred = np.array([3, 2, 3, 0, 1, 2])

        dcg_score = ranking.dcg(y_true, y_pred)
        print(dcg_score)

    def test_ndcg(self):
        y_true = np.array([3, 3, 2, 2, 1, 0])
        y_pred = np.array([3, 2, 3, 0, 1, 2])

        ndcg_score = ranking.ndcg(y_true, y_pred)
        print(ndcg_score)
