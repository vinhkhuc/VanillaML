"""
Root mean square error (RMSE)
"""
import numpy as np


def mse_score(true_y, pred_y):
    return np.mean(np.square(true_y - pred_y))


def rmse_score(true_y, pred_y):
    return np.sqrt(mse_score(true_y, pred_y))
