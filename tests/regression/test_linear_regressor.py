import unittest

from vanilla_ml.regression.linear_regression import LinearRegressor
from vanilla_ml.util import data_io
from vanilla_ml.util.metrics.rmse import rmse_score


class TestLinearRegressor(unittest.TestCase):

    def test_line_with_noises(self):
        train_X, test_X, train_y, test_y = data_io.get_regression_line(noise=True)
        print("train_X's shape = %s, train_y's shape = %s" % (train_X.shape, train_y.shape))
        print("test_X's shape = %s, test_y's shape = %s" % (test_X.shape, test_y.shape))

        regr = LinearRegressor(solver='analytical')
        print("regr: %s" % regr)

        print("Fitting ...")
        regr.fit(train_X, train_y)

        print("Predicting ...")
        pred_y = regr.predict(test_X)
        print("pred_y = %s" % regr.predict(test_X))

        rmse = rmse_score(test_y, pred_y)
        print("RMSE = %g" % rmse)

        # self.assertLessEqual(rmse, 6)

