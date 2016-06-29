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
        self.scaler = None
        self.largest_components = None

    def fit(self, X):
        # Apply standard scaler
        self.scaler = StandardScaler()
        scaled_X = self.scaler.fit_transform(X)
        cov_matrix = np.cov(scaled_X.T)

        # Compute eigenvectors and eigenvalues
        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

        sorted_idx = np.abs(eigen_values).argsort()[::-1]
        self.largest_components = eigen_vectors[:, sorted_idx[:self.n_components]].T
        # print(self.largest_components.T)

    def transform(self, X):
        scaled_X = self.scaler.transform(X)
        return np.dot(scaled_X, self.largest_components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
