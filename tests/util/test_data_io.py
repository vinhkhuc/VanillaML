import unittest
import numpy as np
from vanilla_ml.util import data_io


class TestDataIO(unittest.TestCase):

    def test_train_test_split(self):
        X = np.arange(40).reshape(20, 2)
        y = np.arange(20)
        train_X, test_X, train_y, test_y = data_io.train_test_split(X, y)
        self.assertTrue(train_X.shape == (15, 2))
        self.assertTrue(test_X.shape == (5, 2))
        self.assertTrue(train_y.shape == (15, ))
        self.assertTrue(test_y.shape == (5, ))

    def test_iris_loader(self):
        X, y = data_io.load_iris()
        self.assertTrue(X.shape == (150, 4))
        self.assertTrue(y.shape == (150, ))

