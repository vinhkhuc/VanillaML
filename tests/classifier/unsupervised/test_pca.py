import unittest

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition.pca import PCA as skPCA

from vanilla_ml.classifier.unsupervised.pca import PCA
from vanilla_ml.util import data_io


class TestPCA(unittest.TestCase):

    def test_pca(self):
        X, y = data_io.load_iris()

        print("X's shape = %s" % (X.shape, ))
        pca = PCA(n_components=2)

        print("Fitting ...")
        pca.fit(X)

        pca_X = pca.transform(X)
        print(pca_X)

        unique_labels = np.unique(y)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        for k, col in zip(unique_labels, colors):
            class_member_mask = y == k
            xy = pca_X[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)
        plt.show()
