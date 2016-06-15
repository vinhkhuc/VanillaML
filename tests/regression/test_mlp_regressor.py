import unittest

import numpy as np
from sklearn.preprocessing.data import StandardScaler

from vanilla_ml.regression.abstract_regressor import AbstractRegressor
from vanilla_ml.regression.mlp_regressor import MLPRegressor
from vanilla_ml.util import data_io
from vanilla_ml.util.metrics.rmse import mse_score, rmse_score


class TestMLPRegressor(unittest.TestCase):

    def test_line_with_noises(self):
        # train_X, test_X, train_y, test_y = data_io.get_regression_line(noise=True)
        train_X, test_X, train_y, test_y = data_io.get_regression_line(noise=False)
        print("train_X's shape = %s, train_y's shape = %s" % (train_X.shape, train_y.shape))
        print("test_X's shape = %s, test_y's shape = %s" % (test_X.shape, test_y.shape))

        # # X = [0, 1, 2] and y = [1, 3, 5]
        # X = np.arange(0, 3, 1)
        # train_X = test_X = X[:, None]
        # train_y = test_y = 2 * X + 1

        # train_X = test_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        # train_y = test_y = np.array([0, 1, 1, 0])

        # train_X = test_X = np.array([[0, 0], [0, 1]])
        # train_y = test_y = np.array([0, 1])

        # train_X = test_X = np.array([[0], [1]])
        # train_y = test_y = np.array([0, 1])

        # print("Applying standard scaling ...")
        # scaler = StandardScaler()
        # train_X = scaler.fit_transform(train_X)
        # test_X = scaler.transform(test_X)

        layers = [20]
        learning_rate = 0.01
        batch_size = len(train_X)
        n_rounds = 100
        regr = MLPRegressor(layers, batch_size=batch_size,
                            n_epochs=n_rounds, learning_rate=learning_rate)
        print("regr: %s" % regr)

        print("Fitting ...")
        regr.fit(train_X, train_y)

        print("Predicting ...")
        pred_y = regr.predict(test_X)
        print("pred_y = %s" % regr.predict(test_X))

        rmse = rmse_score(test_y, pred_y)
        print("RMSE = %g" % rmse)

        # self.assertLessEqual(rmse, 6)

if __name__ == '__main__':
    unittest.main()
