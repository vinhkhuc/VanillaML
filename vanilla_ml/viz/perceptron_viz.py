"""
Animated visualization of how Perceptron works
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
    sign_y = misc.label_to_sign(y)
    n_samples, n_features = X.shape
    assert n_features == 2, "Number of features must be 2 for visualization ..."
    w = np.array([-2.0, 3.0])

    # Visualize KMeans
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=[7, 7], facecolor='w')
    ani = animation.FuncAnimation(fig, run_perceptron, range(n_samples),
                                  fargs=(X, y, sign_y, w, ax),
                                  blit=False, interval=1000, repeat=True)
    plt.show()
    ani.save('Perceptron.gif', writer='imagemagick')


def run_perceptron(it, X, y, sign_y, w, ax):
    ax.cla()
    n_samples = X.shape[0]
    i = it % n_samples

    draw_Xy(i, X, y, ax)
    if it == 0:
        draw_separator(it, w, X, ax)

    pred_sign_yi = misc.sign(np.inner(w, X[i]))
    if pred_sign_yi != sign_y[i]:
        w += sign_y[i] * X[i]

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
    # Since w[0] * u + w[1] * v = 0, we have v = -w[0]/w[1] * u
    if w[1] != 0:
        u = np.linspace(-5, 5, 1000)
        v = -w[0]/w[1] * u
        # Filter out the line's parts out of X's ranges
        keep_idx = (min(X[:, 0]) <= u) & (u <= max(X[:, 0])) & \
              (min(X[:, 1]) <= v) & (v <= max(X[:, 1]))
        u = u[keep_idx]
        v = v[keep_idx]
        ax.plot(u, v, '-r')


if __name__ == "__main__":
    viz()
