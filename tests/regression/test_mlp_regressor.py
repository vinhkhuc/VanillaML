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
        # print("train_X's shape = %s, train_y's shape = %s" % (train_X.shape, train_y.shape))
        # print("test_X's shape = %s, test_y's shape = %s" % (test_X.shape, test_y.shape))

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
        regr = MLPRegressor(layers, batch_size=batch_size, n_epochs=n_rounds, learning_rate=learning_rate)
        # regr = KerasRegressor(layer_sizes=layers, batch_size=batch_size, n_rounds=n_rounds, learning_rate=learning_rate)
        print("regr: %s" % regr)

        print("Fitting ...")
        regr.fit(train_X, train_y)

        print("Predicting ...")
        pred_y = regr.predict(test_X)
        print("pred_y = %s" % regr.predict(test_X))

        rmse = rmse_score(test_y, pred_y)
        print("RMSE = %g" % rmse)

        # self.assertLessEqual(rmse, 6)


from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

class KerasRegressor(AbstractRegressor):
    """
    Keras-based regressor.
    """
    def __init__(self, layer_sizes, n_rounds=100, batch_size=64, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.n_rounds = n_rounds
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = None
        self.model = None

    def fit(self, X, y):

        X = np.hstack((X, np.ones((X.shape[0], 1))))

        print("Building model ...")
        self.input_dim = X.shape[1]
        model = Sequential()
        for k, layer_size in enumerate(self.layer_sizes):
            if k == 0:
                model.add(Dense(layer_size,
                                input_shape=(self.input_dim,),
                                init='he_normal'))
            else:
                model.add(Dense(layer_size, init='he_normal'))
            # model.add(PReLU())
            model.add(Activation('sigmoid'))
            model.add(BatchNormalization())
            model.add(Dropout(0.1))

        model.add(Dense(1))
        model.add(Activation('linear'))
        # optimizer = SGD(lr=self.learning_rate)
        # optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        optimizer = 'adagrad'

        model.compile(loss='mse', optimizer=optimizer)

        print("Fitting model ...")
        model.fit(X, y, nb_epoch=self.n_rounds, shuffle=True,
                  batch_size=self.batch_size, validation_split=0.15)
        self.model = model

    def predict(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1))))

        assert self.input_dim == X.shape[1], "Input dimension between training and test set mismatch"
        return self.model.predict(X)


if __name__ == '__main__':
    unittest.main()
