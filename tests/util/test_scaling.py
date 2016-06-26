import unittest

import numpy as np

from vanilla_ml.util.scaling.standard_scaler import StandardScaler


class TestScaling(unittest.TestCase):

    def test_standard_scaling(self):
        X = np.array([[1., 3., 5.], [2., 4., 6.]])
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X)
        self.assertTrue(np.allclose(scaled_X.mean(axis=0), 0))
        self.assertTrue(np.allclose(scaled_X.std(axis=0), 1))
