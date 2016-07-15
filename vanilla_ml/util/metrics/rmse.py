"""
Root mean square error (RMSE)
"""
import numpy as np


def mse_score(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def rmse_score(y_true, y_pred):
    return np.sqrt(mse_score(y_true, y_pred))
