"""
ROC curve (Receiver Operating Characteristic) is the plot of the fraction
of true positives out of the positives (i.e. true positive rate) vs. the
fraction of false positives out of the negatives (i.e. false positive rate).

ROC score is the area under the curve (AUC).

"""
import metric_common


def roc_score(y_true, y_pred_proba):
    """ ROC score (AUC).

    Args:
        y_true (ndarray): shape N
        y_pred_proba (ndarray): predicted probability, shape N

    Returns:
        float: roc score

    """
    assert metric_common.check_binary(y_true) and metric_common.check_range(y_pred_proba, 0, 1), \
        "y_true must be from {0, 1} and pred_proba_y must be in the range [0, 1]."

    raise NotImplemented("To be implemented :(")
