"""
Principal Component Analysis (PCA) is a method used to explain the maximum amount of
variance of a data set using the fewest number of uncorrelated variables which are called
"principal components".
"""
import numpy as np
from vanilla_ml.util.scaling.standard_scaler import StandardScaler


class PCA(object):

    def __init__(self, n_components=3):
        self.n_components = n_components
        self.largest_components = None

    def fit(self, X):
        # Apply standard scaler
        scaled_X = StandardScaler().fit_transform(X)
        # scaled_X = X - X.mean(axis=0)
        cov_matrix = np.cov(scaled_X.T)

        # Compute eigenvectors and eigenvalues
        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

        sorted_idx = np.abs(eigen_values).argsort()[::-1]
        self.largest_components = eigen_vectors[:, sorted_idx[:self.n_components]].T
        # print(self.largest_components.T)

    def transform(self, X):
        return np.dot(X, self.largest_components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
