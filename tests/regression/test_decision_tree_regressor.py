import unittest

import numpy as np
from vanilla_ml.regression.decision_tree_regressor import DecisionTreeRegressor
from vanilla_ml.util import data_io

from sklearn.tree.tree import DecisionTreeRegressor as skDecisionTreeRegressor

class TestDecisionTreeClassifier(unittest.TestCase):
    def test_two_moons(self):
        train_X, test_X, train_y, test_y = data_io.get_moons_train_test(num_samples=200)
        print("train_X's shape = %s, train_y's shape = %s" % (train_X.shape, train_y.shape))
        print("test_X's shape = %s, test_y's shape = %s" % (test_X.shape, test_y.shape))

        # regr = DecisionTreeRegressor(max_depth=5, criterion='mse')
        regr = skDecisionTreeRegressor(max_depth=5, criterion='mse')
        print("regr: %s" % regr)

        print("Fitting ...")
        regr.fit(train_X, train_y)

        print("Predicting ...")
        pred_y = regr.predict(test_X)
        print("pred_y = %s" % regr.predict(test_X))

        rmse = np.sqrt(np.mean(np.square(test_y - pred_y)))
        print("RMSE = %g" % rmse)

        self.assertLessEqual(rmse, 0.5)

if __name__ == '__main__':
    unittest.main()

