"""
Animated visualization of how boosted regression tree works
"""
import numpy as np
# import matplotlib
# matplotlib.rcParams["animation.convert_path"] = "Full path to imagemagic's convert.exe"
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from vanilla_ml.supervised.regression.gradient_boosted_regressor import GradientBoostedRegressor

# FIXME: After iteration 20 (?), nothing changes
# FIXME: Linear Regression doesn't work with GradientBoosting.
from vanilla_ml.supervised.regression.knn_regressor import KNNRegressor
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
    # ani.save('BoostedRegressor.gif', writer='imagemagick')


def run_boosted_regression_tree(i, X, train_y, test_y, ax):

    base_regr = KNNRegressor(k=3)
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

if __name__ == "__main__":
    viz()
