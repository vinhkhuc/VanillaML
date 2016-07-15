
def accuracy_score(y_true, y_pred):
    """ Compute accuracy

    Args:
        y_true (ndarray): ground truth labels
        y_pred (ndarray): predicted labels

    Returns:
        float: accuracy score.

    """
    return (y_true == y_pred).mean()
