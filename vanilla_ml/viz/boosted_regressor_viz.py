"""
Animated visualization of how boosted regression tree works
"""
import numpy as np
# import matplotlib
# matplotlib.rcParams["animation.convert_path"] = "Full path to imagemagic's convert.exe"
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from vanilla_ml.regression.decision_tree_regressor import DecisionTreeRegressor
from vanilla_ml.regression.gradient_boosted_regressor import GradientBoostedRegressor

# FIXME: After iteration 20 (?), nothing changes
# FIXME: Linear Regression doesn't work with GradientBoosting.
from vanilla_ml.regression.knn_regressor import KNNRegressor
from vanilla_ml.regression.linear_regression import LinearRegressor
from vanilla_ml.regression.mlp_regressor import MLPRegressor
from vanilla_ml.util.metrics.rmse import rmse_score


def viz():
    # Prepare data
    X = np.arange(0, 10, 0.1)
    e = np.random.normal(0, 1, size=len(X))
    # test_y = X * np.sin(np.square(X))
    test_y = X * np.sin(X)
    train_y = test_y + e
    X = X[:, None]
    print("X's shape = %s, train_y's shape = %s, test_y's shape = %s"
          % (X.shape, train_y.shape, test_y.shape))

    # Visualize
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=[10, 10], facecolor='w')
    ani = animation.FuncAnimation(fig, run_boosted_regression_tree, range(300),
                                  fargs=(X, train_y, test_y, ax),
                                  blit=False, interval=2000, repeat=True)
    plt.show()
    # ani.save('KMeans.gif', writer='imagemagick')


def run_boosted_regression_tree(i, X, train_y, test_y, ax):

    # base_regr = DecisionTreeRegressor(max_depth=1)
    # base_regr = LinearRegressor(solver='analytical')  # boosted linear regression will be a line !!!
    # base_regr = MLPRegressor(layers=[200], batch_size=len(X), n_epochs=500, learning_rate=0.1)
    base_regr = KNNRegressor(k=11)
    # base_regr = KerasRegressor(layer_sizes=[2], batch_size=100, n_rounds=10)
    regr = GradientBoostedRegressor(base_regr, num_rounds=i + 1, alpha=0.1)

    print("\n* Iteration = %d" % (i + 1))
    print("Fitting ...")
    regr.fit(X, train_y)

    print("Predicting ...")
    pred_y = regr.predict(X)
    # print("pred_y[0] = %g" % pred_y[0])
    train_rmse = rmse_score(train_y, pred_y)
    test_rmse = rmse_score(test_y, pred_y)
    print("Train RMSE = %g, test RMSE = %g" % (train_rmse, test_rmse))

    ax.cla()
    ax.set_ylim([train_y.min() - 1, train_y.max() + 1])
    ax.plot(X, train_y, 'r')
    ax.plot(X, test_y, 'g')
    ax.plot(X, pred_y, 'b')
    ax.set_title('Iteration %d, training RMSE = %g, test RMSE = %g' % (i + 1, train_rmse, test_rmse))

from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization

class KerasRegressor(object):
    """
    Keras-based regressor.
    """
    def __init__(self, layer_sizes, n_rounds=100, batch_size=64):
        self.layer_sizes = layer_sizes
        self.n_rounds = n_rounds
        self.batch_size = batch_size
        self.input_dim = None
        self.model = None

    def fit(self, X, y):
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
        model.compile(loss='mse', optimizer='adagrad')

        print("Fitting model ...")
        model.fit(X, y, nb_epoch=self.n_rounds, shuffle=True,
                  batch_size=self.batch_size, validation_split=0.15)
        self.model = model

    def predict(self, X):
        assert self.input_dim == X.shape[1], "Input dimension between training and test set mismatch"
        return self.model.predict(X).squeeze()

if __name__ == "__main__":
    viz()
