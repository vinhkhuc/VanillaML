"""
Root mean square error (RMSE)
"""
import numpy as np


def rmse_score(true_y, pred_y):
    return np.sqrt(np.mean(np.square(true_y - pred_y)))
