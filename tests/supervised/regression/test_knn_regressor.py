import unittest

from vanilla_ml.supervised.regression.knn_regressor import KNNRegressor
from vanilla_ml.util import data_io
from vanilla_ml.util.metrics.rmse import rmse_score
from vanilla_ml.util.scaling.standard_scaler import StandardScaler


class TestKNNRegressor(unittest.TestCase):

    def test_line_with_noises(self):
        train_X, test_X, train_y, test_y = data_io.get_regression_line(noise=True)
        # train_X, test_X, train_y, test_y = data_io.get_regression_line(noise=False)
        print("train_X's shape = %s, train_y's shape = %s" % (train_X.shape, train_y.shape))
        print("test_X's shape = %s, test_y's shape = %s" % (test_X.shape, test_y.shape))

        print("Applying standard scaling ...")
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

        regr = KNNRegressor(k=3)
        # regr = skKNNRegressor(n_neighbors=3)

        print("Fitting ...")
        regr.fit(train_X, train_y)

        print("Predicting ...")
        pred_y = regr.predict(test_X)
        print("pred_y = %s" % pred_y)

        rmse = rmse_score(test_y, pred_y)
        print("RMSE = %g" % rmse)

        # self.assertLessEqual(rmse, 6)
