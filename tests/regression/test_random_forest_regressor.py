import unittest

import numpy as np

from vanilla_ml.regression.random_forest_regressor import RandomForestRegressor
from vanilla_ml.util import data_io


class TestRandomForestRegressor(unittest.TestCase):

    def test_boston(self):
        train_X, test_X, train_y, test_y = data_io.get_boston_train_test()
        print("train_X's shape = %s, train_y's shape = %s" % (train_X.shape, train_y.shape))
        print("test_X's shape = %s, test_y's shape = %s" % (test_X.shape, test_y.shape))

        clf = RandomForestRegressor(num_trees=5, max_depth=3, criterion='mse')
        print("clf: %s" % clf)

        print("Fitting ...")
        clf.fit(train_X, train_y)

        print("Predicting ...")
        pred_y = clf.predict(test_X)
        print("pred_y = %s" % clf.predict(test_X))

        rmse = np.sqrt(np.mean(np.square(test_y - pred_y)))
        print("RMSE = %g" % rmse)

        self.assertLessEqual(rmse, 6)

if __name__ == '__main__':
    unittest.main()
