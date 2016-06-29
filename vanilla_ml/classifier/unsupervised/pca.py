"""
Principal Component Analysis (PCA) is a method used to explain the maximum amount of
variance of a data set using the fewest number of uncorrelated variables which are called
"principal components".
"""
import numpy as np


class PCA(object):

    def __init__(self, n_components=3):
        self.n_components = n_components
        self.mean = None
        self.largest_components = None

    def fit(self, X):
        # Center the data
        self.mean = X.mean(axis=0)
        scaled_X = X - self.mean

        # The returned eigenvectors are sorted in the decreasing order of their corresponding eigenvalues.
        U, S, V = np.linalg.svd(scaled_X, full_matrices=False)
        self.largest_components = V[:self.n_components]

    def transform(self, X):
        scaled_X = X - self.mean
        return np.dot(scaled_X, self.largest_components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
