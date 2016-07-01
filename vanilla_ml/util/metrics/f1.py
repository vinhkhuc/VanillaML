"""
Compute F1 score

    F1 = 2 * P * R / (P + R)
    where P = TP / (TP + FP), R = TP / (TP + FN)

Ref: https://www.kaggle.com/wiki/Metrics

"""
import numpy as np

import metric_common


def f1_score(true_y, pred_y):
    """ F1 score

    Args:
        true_y (ndarray): shape N
        pred_y (ndarray): shape N

    Returns:
        float: f1 score

    """
    assert metric_common.check_binary(true_y) and metric_common.check_binary(pred_y), \
        "Only binary arrays are supported."

    true_y = true_y.squeeze()
    pred_y = pred_y.squeeze()

    p = precision(true_y, pred_y)
    r = recall(true_y, pred_y)
    return 2 * p * r / (p + r)


def precision(true_y, pred_y):
    """ Precision

    Args:
        true_y (ndarray): shape N
        pred_y (ndarray): shape N

    Returns:
        float: precision score

    """
    assert metric_common.check_binary(true_y) and metric_common.check_binary(pred_y), \
        "Only binary arrays are supported."

    true_y = true_y.squeeze()
    pred_y = pred_y.squeeze()

    tp = np.sum(true_y == pred_y)
    fp = np.sum((pred_y == 1) & (true_y == 0))
    return float(tp) / (tp + fp)


def recall(true_y, pred_y):
    """ Recall

    Args:
        true_y (ndarray): shape N
        pred_y (ndarray): shape N

    Returns:
        float: recall score

    """
    assert metric_common.check_binary(true_y) and metric_common.check_binary(pred_y), \
        "Only binary arrays are supported."

    true_y = true_y.squeeze()
    pred_y = pred_y.squeeze()

    tp = np.sum(pred_y == true_y)
    fn = np.sum((pred_y == 0) & (true_y == 1))
    return float(tp) / (tp + fn)
