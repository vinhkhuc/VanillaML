"""
Animated visualization of how boosted regression tree works
"""
import numpy as np

from vanilla_ml.supervised.regression.decision_tree_regressor import DecisionTreeRegressor

# import matplotlib
# matplotlib.rcParams["animation.convert_path"] = "Full path to ImageMagick's convert.exe"
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from vanilla_ml.supervised.regression.gradient_boosted_regressor import GradientBoostedRegressor
from vanilla_ml.util.metrics.rmse import rmse_score


def viz():
    # Prepare data
    X = np.arange(0, 10, 0.1)
    test_y = X * np.sin(X)
    # e = np.random.normal(0, 1, size=len(X))
    # train_y = test_y + e
    train_y = test_y.copy()
    X = X[:, None]
    print("X's shape = %s, train_y's shape = %s, test_y's shape = %s"
          % (X.shape, train_y.shape, test_y.shape))

    # Visualize
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=[7, 7], facecolor='w')
    ani = animation.FuncAnimation(fig, run_boosted_regression_tree, range(25),
                                  fargs=(X, train_y, test_y, ax),
                                  blit=False, interval=1000, repeat=True)
    plt.show()
    # ani.save('BoostedRegressor.gif', writer='imagemagick')


def run_boosted_regression_tree(i, X, train_y, test_y, ax):

    base_regr = DecisionTreeRegressor()
    regr = GradientBoostedRegressor(base_regr, num_rounds=i + 1, alpha=0.1)

    print("\n* Iteration = %d" % (i + 1))
    print("Fitting ...")
    regr.fit(X, train_y)

    print("Predicting ...")
    pred_y = regr.predict(X)
    train_rmse = rmse_score(train_y, pred_y)
    test_rmse = rmse_score(test_y, pred_y)
    # print("Train RMSE = %g, test RMSE = %g" % (train_rmse, test_rmse))
    print("Train RMSE = %g" % train_rmse)

    ax.cla()
    ax.set_ylim([train_y.min() - 1, train_y.max() + 1])
    ax.plot(X, train_y, 'r')
    ax.plot(X, test_y, 'g')
    ax.plot(X, pred_y, 'b')
    # ax.set_title('Iteration %d, training RMSE = %g, test RMSE = %g' % (i + 1, train_rmse, test_rmse))
    ax.set_title('Iteration %d, training RMSE = %g' % (i + 1, train_rmse))

if __name__ == "__main__":
    viz()
