import unittest

from vanilla_ml.classifier.unsupervised.kmeans import KMeans
from vanilla_ml.util import data_io
from vanilla_ml.util.metrics.cluster_purity import cluster_purity


class TestKMeans(unittest.TestCase):

    def test_kmeans(self):
        centers = [[1, 1], [-1, -1], [1, -1]]
        n_clusters = len(centers)
        X, y = data_io.get_clustering_data(centers=centers, n_samples=50, cluster_std=0.7)
        print("X's shape = %s, y's shape = %s" % (X.shape, y.shape))

        kmeans = KMeans(n_clusters=n_clusters)

        print("Fitting ...")
        kmeans.fit(X)

        print("Predicting ...")
        pred_y = kmeans.predict(X)
        print("y = %s" % y)
        print("pred_y = %s" % pred_y)

        purity = cluster_purity(y, pred_y)
        print("Cluster purity = %.2f%%" % (100. * purity))
        self.assertGreaterEqual(purity, 0.9)
