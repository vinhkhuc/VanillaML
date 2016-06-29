"""
Compute F1 score

    F1 = 2 * P * R / (P + R)
    where P = TP / (TP + FP), R = TP / (TP + FN)

Ref: https://www.kaggle.com/wiki/Metrics

"""

import numpy as np


def f1_score(pred_y, true_y):
    """ F1 score

    Args:
        pred_y (ndarray): shape N
        true_y (ndarray): shape N

    Returns:
        float: f1 score

    """
    assert _check_binary(pred_y) and _check_binary(true_y), "Only binary arrays are supported."

    pred_y = pred_y.squeeze()
    true_y = true_y.squeeze()

    p = precision(pred_y, true_y)
    r = recall(pred_y, true_y)
    return 2 * p * r / (p + r)


def precision(pred_y, true_y):
    """ Precision

    Args:
        pred_y (ndarray): shape N
        true_y (ndarray): shape N

    Returns:
        float: precision score

    """
    assert _check_binary(pred_y) and _check_binary(true_y), "Only binary arrays are supported."

    pred_y = pred_y.squeeze()
    true_y = true_y.squeeze()

    tp = np.sum(pred_y == true_y)
    fp = np.sum((pred_y == 1) & (true_y == 0))
    return float(tp) / (tp + fp)


def recall(pred_y, true_y):
    """ Recall

    Args:
        pred_y (ndarray): shape N
        true_y (ndarray): shape N

    Returns:
        float: recall score

    """
    assert _check_binary(pred_y) and _check_binary(true_y), "Only binary arrays are supported."

    pred_y = pred_y.squeeze()
    true_y = true_y.squeeze()

    tp = np.sum(pred_y == true_y)
    fn = np.sum((pred_y == 0) & (true_y == 1))
    return float(tp) / (tp + fn)


def _check_binary(x):
    return np.all((x == 0) | (x == 1))
