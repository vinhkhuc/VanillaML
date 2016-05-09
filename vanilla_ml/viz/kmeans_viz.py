"""
Animated visualization of how KMeans works
"""
import numpy as np
# import matplotlib
# matplotlib.rcParams["animation.convert_path"] = "Full path to imagemagic's convert.exe"
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from vanilla_ml.util import data_io
from vanilla_ml.util.distances import compute_dist_matrix


def viz():
    # Prepare data
    centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
    n_clusters = len(centers)
    X, _ = data_io.get_clustering_data(centers=centers, n_samples=1000, cluster_std=0.5)
    print("X's shape = %s" % (X.shape,))

    # Visualize KMeans
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=[10, 10], facecolor='w')
    ani = animation.FuncAnimation(fig, run_kmeans, range(8), fargs=(X, n_clusters, ax),
                                  blit=False, interval=2000, repeat=True)
    plt.show()
    # ani.save('KMeans.gif', writer='imagemagick')


def run_kmeans(i, X, n_clusters, ax):
    np.random.seed(42)
    n_samples, n_features = X.shape

    cluster_centroids = np.zeros((n_clusters, n_features), dtype=float)
    cluster_sizes = np.empty(n_clusters)

    # Assign each sample to a random cluster
    y = np.random.randint(0, n_clusters, size=n_samples)

    if i == 0:
        draw_plot(X, y, i, ax)

    for it in range(i):
        # Calculate cluster centroids
        cluster_sizes.fill(0)
        for x_i, y_i in zip(X, y):
            cluster_centroids[y_i] += x_i
            cluster_sizes[y_i] += 1
        cluster_centroids /= cluster_sizes[:, None]

        # Reassign samples to new clusters
        dist_matrix = compute_dist_matrix(X, cluster_centroids, distance='l2')
        next_y = dist_matrix.argmin(axis=1)

        # Check if clusters have changed
        if (y == next_y).all():
            break

        # Otherwise continue
        y = next_y

    draw_plot(X, y, i, ax)


def draw_plot(X, y, it, ax):
    unique_labels = set(y)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'k'  # black color is used for noise
        class_member_mask = y == k
        xy = X[class_member_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=10)
    ax.set_title('Iteration %d' % (it + 1))

if __name__ == "__main__":
    viz()
