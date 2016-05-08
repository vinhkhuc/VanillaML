"""
Cluster purity:
"""
import numpy as np


def cluster_purity(true_classes, pred_clusters):
    """ Compute cluster purity.
    Ref: http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html

    Args:
        true_classes (ndarray): true classes, shape N x 1.
        pred_clusters (ndarray): predicted clusters, shape N x 1.

    Returns:
        float: cluster purity score

    """
    assert len(true_classes) == len(pred_clusters), \
        "Number of true classes and number of predicted clusters mismatch"

    n_samples = len(true_classes)
    n_clusters = len(np.unique(pred_clusters))

    count = 0
    for k in range(n_clusters):
        classes_by_cluster = true_classes[pred_clusters == k]
        class_freq_by_cluster = np.bincount(classes_by_cluster)
        count += class_freq_by_cluster.max()
    return count / float(n_samples)
