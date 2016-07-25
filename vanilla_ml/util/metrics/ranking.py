import numpy as np


def dcg(y_true, y_pred, k=10):
    """ Compute Discounted Cumulative Gain (DCG).
    See: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain

    Args:
        y_true (ndarray): array of true relevant scores, shape N.
        y_pred (ndarray): array of predicted relevant scores, shape N.
        k (Optional[int]):

    Returns:
        float: dcg@k score

    """
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    assert len(y_true) == len(y_pred), "Lengths of y_true and y_pred mismatch!"
    N = min(k, len(y_true))

    sorted_indices = np.argsort(y_pred)[::-1][:N]
    exp_relevances = 2 ** y_true[sorted_indices] - 1
    log_positions = np.log2(np.arange(N) + 2)

    return sum(exp_relevances / log_positions)


def idcg(y_true, k=10):
    """ Compute Ideal Discounted Cumulative Gain (IDCG).
    See: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    Args:
        y_true (ndarray): array of true relevant scores, shape N
        k (Optional[int]):

    Returns:
        float: idcg@k score

    """
    return dcg(y_true, y_true, k)


def ndcg(y_true, y_pred, k=10):
    """ Compute Normalized Discounted Cumulative Gain (NDCG).
    See: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    Args:
        y_true (ndarray): array of true relevant scores, shape N.
        y_pred (ndarray): array of predicted relevant scores, shape N.
        k (Optional[int]):

    Returns:
        float: ndcg@k score

    """
    dcg_score = dcg(y_true, y_pred, k)
    ideal_dcg_score = idcg(y_true, k)
    return dcg_score / ideal_dcg_score


def dcg_at_ix(y_true, true_idx, pred_idx):
    """ Compute DCG at a true ranked index given its corresponding predicted ranked index.

    Args:
        y_true (ndarray): array of true relevant scores, shape N.
        true_idx (int): true ranking index.
        pred_idx (int): predicted ranking index.

    Returns:
        float: dcg score at specified ranking index

    """
    return (2 ** y_true[true_idx] - 1) / np.log2(pred_idx + 2)


def delta_dcg(y_true, y_pred, i, j):
    """ Compute the DCG delta if we swap two ranked indices i and j.

    Args:
        y_true (ndarray): array of true relevant scores, shape N.
        y_pred (ndarray): array of predicted relevant scores, shape N.
        i (int): ranking index.
        j (int): ranking index.

    Returns:
        float: DCG delta score.

    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    sorted_indices = np.argsort(y_pred)[::-1]  # TODO: Need to cache it somewhere
    idx_i, idx_j = sorted_indices[i], sorted_indices[j]
    return dcg_at_ix(y_true, idx_i, i) + dcg_at_ix(y_true, idx_j, j) \
           - dcg_at_ix(y_true, idx_i, j) - dcg_at_ix(y_true, idx_j, i)


def delta_ndcg(y_true, y_pred, i, j):
    """ Compute the NDCG delta if we swap two ranked indices i and j.

    Args:
        y_true (ndarray): array of true relevant scores, shape N.
        y_pred (ndarray): array of predicted relevant scores, shape N.
        i (int): ranking index.
        j (int): ranking index.

    Returns:
        float: NDCG delta score.

    """
    delta_dcg_score = delta_dcg(y_true, y_pred, i, j)
    idcg_score = idcg(y_true, k=len(y_true))
    return delta_dcg_score / idcg_score

    # k = len(y_true)
    # orig_ndcg = ndcg(y_true, y_pred, k=k)
    #
    # swapped_y_pred = np.copy(y_pred)
    # swapped_y_pred[i] = y_pred[j]
    # swapped_y_pred[j] = y_pred[i]
    # swapped_ndcg = ndcg(y_true, swapped_y_pred, k=k)
    #
    # return orig_ndcg - swapped_ndcg
