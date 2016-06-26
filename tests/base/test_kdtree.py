import unittest

import numpy as np
from scipy.spatial.kdtree import KDTree

from vanilla_ml.base.kdtree import KDTree


class TestKDTree(unittest.TestCase):

    def test_one_nearest_neighbor(self):
        X = np.array([[1, 6], [2, 2], [3, 7], [5, 4], [6, 8], [6, 1], [7, 5]])
        kd_tree = KDTree()

        print("Building KDTree ...")
        kd_tree.build(X)

        print("Find nearest neighbor ...")
        test_X = np.array([[3, 5], [4.5, 2]])
        print(kd_tree.find_nearest(test_X))
