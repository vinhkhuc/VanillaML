import numpy as np


def dcg(y_true, y_pred, k=10):
    """ Compute Discounted Cumulative Gain (DCG).
    See: https://en.wikipedia.org/wiki/Discounted_cumulative_gain

    Args:
        y_true (ndarray): array of true relevant scores, shape N
        y_pred (ndarray): array of predicted relevant scores, shape N
        k (Optional[int]):

    Returns:
        float: dcg @k score

    """
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    assert len(y_true) == len(y_pred), "Lengths of y_true and y_pred mismatch!"
    N = min(k, len(y_true))

    sorted_indices = np.argsort(y_pred)[::-1][:N]
    exp_relevances = 2 ** y_true[sorted_indices] - 1
    log_positions = np.log2(np.arange(N) + 2)

    return sum(exp_relevances / log_positions)


def ndcg(y_true, y_pred, k=10):
    """ Compute Normalized Discounted Cumulative Gain (NDCG).
    See: https://en.wikipedia.org/wiki/Discounted_cumulative_gain

    Args:
        y_true (ndarray): array of true relevant scores, shape N
        y_pred (ndarray): array of predicted relevant scores, shape N
        k (Optional[int]):

    Returns:
        float: ndcg @k score

    """
    dcg_score = dcg(y_true, y_pred, k)
    ideal_dcg_score = dcg(y_true, y_true, k)
    return dcg_score / ideal_dcg_score
