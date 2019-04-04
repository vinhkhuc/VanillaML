"""
Animated visualization of how Logistic Regression works
"""
import numpy as np
# import matplotlib
# matplotlib.rcParams["animation.convert_path"] = "Full path to ImageMagick's convert.exe"
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from vanilla_ml.util import data_io, misc


def viz():
    # Prepare data
    X, y = data_io.get_binary_classification_data(n_samples=15, cluster_std=0.4)
    one_hot_y = np.eye(2)[y]
    n_samples, n_features = X.shape
    assert n_features == 2, "Number of features must be 2 for visualization ..."
    w = np.array([-2.0, 3.0])

    # Visualize KMeans
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=[7, 7], facecolor='w')
    ani = animation.FuncAnimation(fig, run_linear_svm, range(5 * n_samples),
                                  fargs=(X, y, one_hot_y, w, ax),
                                  blit=False, interval=1000, repeat=True)
    plt.show()
    ani.save('LogisticRegression.gif', writer='imagemagick')


def run_linear_svm(it, X, y, onehot_y, w, ax):
    ax.cla()
    n_samples = X.shape[0]
    i = it % n_samples

    draw_Xy(i, X, y, ax)
    # if it == 0:
    #     draw_separator(it, w, X, ax)

    # pred_sign_yi = misc.sign(np.inner(w, X[i]))
    # if pred_sign_yi != sign_y[i]:
    #     w += sign_y[i] * X[i]

    # TODO: Move these into constructor
    lr = 0.05
    penalty_factor = 1.0

    # yX_i = X[i] * sign_y[i]
    # grad = -yX_i if np.inner(w, yX_i) <= 1 else 0
    # grad += 2 * penalty_factor * w
    # w -= lr * grad

    # sigmoid_Xw = misc.sigmoid(np.dot(X[i], w))
    # pred_proba_y_i = np.array([sigmoid_Xw, 1 - sigmoid_Xw])
    # pred_proba_y_i = 1 - pred_proba_y_i # DEBUG ONLY!!!
    # grad = np.dot(pred_proba_y_i - onehot_y[i], X[i])

    pred_proba_y_i = misc.sigmoid(np.dot(X[i], w))
    # grad = np.dot(pred_proba_y_i - y[i], X[i])
    grad = (pred_proba_y_i - y[i]) * X[i]

    grad += 2 * penalty_factor * w
    w -= lr * grad

    print(i, pred_proba_y_i, onehot_y[i])
    draw_separator(it, w, X, ax)


def draw_Xy(i, X, y, ax):
    unique_labels = np.unique(y)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'k'  # black color is used for noise
        class_member_mask = y == k
        xy = X[class_member_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)

    # Highlight active example
    ax.plot(X[i, 0], X[i, 1], 'o', markerfacecolor='g', markeredgecolor='y', markersize=14)


def draw_separator(it, w, X, ax):
    ax.set_title('Iteration %d' % (it + 1))
    # # Since w[0] * u + w[1] * v = 0, we have v = -w[0]/w[1] * u
    # if w[1] != 0:
    #     u = np.linspace(-5, 5, 1000)
    #     v = -w[0]/w[1] * u
    #     # Filter out the line's parts out of X's ranges
    #     keep_idx = (min(X[:, 0]) <= u) & (u <= max(X[:, 0])) & \
    #           (min(X[:, 1]) <= v) & (v <= max(X[:, 1]))
    #     u = u[keep_idx]
    #     v = v[keep_idx]
    #     ax.plot(u, v, '-r')

    xx_min, xx_max = min(X[:, 0]), max(X[:, 0])
    yy_min, yy_max = min(X[:, 1]), max(X[:, 1])
    xx, yy = np.mgrid[xx_min:xx_max:.01, yy_min:yy_max:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = misc.sigmoid(np.dot(grid, w)).reshape(xx.shape)

    ax.contourf(xx, yy, probs, 25, cmap="RdBu", vmin=0, vmax=1)
    # ax.contourf(xx, yy, probs, 10, cmap="RdBu", vmin=0, vmax=1)
    # ax.contour(xx, yy, probs, levels=[.5], cmap="Reds", vmin=0, vmax=.6)

    # f, ax = plt.subplots(figsize=(8, 6))
    # contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu", vmin=0, vmax=1)
    # ax_c = f.colorbar(contour)
    # ax_c.set_label("$P(y = 1)$")
    # ax_c.set_ticks([0, .25, .5, .75, 1])

    # ax.scatter(X[100:, 0], X[100:, 1], c=y[100:], s=50,
    #            cmap="RdBu", vmin=-.2, vmax=1.2,
    #            edgecolor="white", linewidth=1)
    #
    # ax.set(aspect="equal", xlim=(-5, 5), ylim=(-5, 5),
    #        xlabel="$X_1$", ylabel="$X_2$")


if __name__ == "__main__":
    viz()
