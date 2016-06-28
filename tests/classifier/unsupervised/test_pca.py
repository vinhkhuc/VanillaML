import unittest

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition.pca import PCA as skPCA

from vanilla_ml.classifier.unsupervised.pca import PCA
from vanilla_ml.util import data_io


# FIXME: The results are different from sklearn's PCA
class TestPCA(unittest.TestCase):

    def test_pca(self):
        X, y = data_io.load_iris()
        # import pandas as pd
        # df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
        #                       header=None)
        # from sklearn.cross_validation import train_test_split
        # from sklearn.preprocessing import StandardScaler
        # X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        # sc = StandardScaler()
        # X_train_std = sc.fit_transform(X_train)
        # X_test_std = sc.transform(X_test)
        # X = X_train_std

        print("X's shape = %s" % (X.shape, ))

        # pca = PCA(n_components=2)
        pca = skPCA(n_components=2)

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
