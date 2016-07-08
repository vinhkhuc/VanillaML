"""
ROC curve (Receiver Operating Characteristic) is the plot of the fraction
of true positives out of the positives (i.e. true positive rate) vs. the
fraction of false positives out of the negatives (i.e. false positive rate).

ROC score is the area under the curve (AUC).

"""
import metric_common


def roc_score(true_y, pred_proba_y):
    """ ROC score (AUC).

    Args:
        true_y (ndarray): shape N
        pred_proba_y (ndarray): predicted probability, shape N

    Returns:
        float: roc score

    """
    assert metric_common.check_binary(true_y) and metric_common.check_range(pred_proba_y, 0, 1), \
        "true_y must be from {0, 1} and pred_proba_y must be in the range [0, 1]."

    raise NotImplemented("To be implemented :(")
