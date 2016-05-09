
def accuracy_score(true_y, pred_y):
    """ Compute accuracy

    Args:
        true_y (ndarray): ground truth labels
        pred_y (ndarray): predicted labels

    Returns:
        float: accuracy score.

    """
    return (true_y == pred_y).mean()
