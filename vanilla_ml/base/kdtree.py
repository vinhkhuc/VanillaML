from __future__ import division

import numpy as np

from vanilla_ml.util import distances


# TODO: The current implementation only support finding the nearest neighbors (i.e. k=1)
class KDTree(object):
    """
    KDTree (K-Dimensional Tree) was invented Jon L. Bentley. KDTree is a binary
    tree constructed by recursively splitting at the median of data points along
    a dimension in an alternative or randomized fashion.

    Complexity:
        + Construction time complexity = O(N * log^2 N)
        + Query time complexity        = O(logN)
        + Space complexity             = O(N)
        where N is the number of data points

    """
    def __init__(self):
        self.root = None

    def build(self, X, y=None):
        self.root = _build_recursively(X, y, depth=0)

    def find_nearest(self, X, k=1):

        if k > 1:
            raise Exception("Only K=1 is supported :(")

        assert X.ndim == 2, "X must be a list of data points."
        return [_find_nearest(X_i, self.root, k=k, depth=0) for X_i in X]


def _build_recursively(X, y, depth):
    """ Recursively build tree's node at each depth.

    Args:
        X (ndarray): data points, shape N x P.
        y (ndarray): data labels, shape N x 1.
        depth (int): current depth

    Returns:
        TreeNode: constructed tree's node.
    """

    n_samples, n_dimensions = X.shape

    if n_samples == 0:
        return None

    # Alternatively pick a dimension to split on
    split_axis = _get_split_axis(depth, n_dimensions)

    # Sort
    sorted_indices = np.argsort(X[:, split_axis])
    sorted_X = X[sorted_indices]

    # Get the index of the median element
    median_idx = int((n_samples - 1) / 2)  # due to 0-indexing (6 -> 2, 7 -> 2)
    left_X, right_X = sorted_X[:median_idx], sorted_X[(median_idx + 1):]
    median_X = sorted_X[median_idx].squeeze()

    if y is not None:
        sorted_y = y[sorted_indices]
        median_y = sorted_y[median_idx]
        left_y, right_y = sorted_y[:median_idx], sorted_y[(median_idx + 1):]
    else:
        median_y = None
        left_y, right_y = None, None

    # Recursively split
    node_val = (median_X, median_y)
    node = TreeNode(node_val)
    node.left = _build_recursively(left_X, left_y, depth + 1)
    node.right = _build_recursively(right_X, right_y, depth + 1)

    return node


def _find_nearest(X_i, node, k, depth):
    """ Find the node containing the data point nearest to the given data point.

    Args:
        X_i (ndarray): single input data point, shape P.
        node (TreeNode): tree's node at the current depth.
        k (int): number of nearest neighbors.
        depth (int): current depth.

    Returns:
        tuple: a tuple of (nearest node, distance to the nearest node, height from the leaf node).

    """
    assert X_i.ndim == 1, "The input data point must be a 1-D array."

    # Current node's data point
    node_X = node.val[0]

    # Check if we have reached the leaf node.
    # If node's left subtree is None, so is the right one.
    if node.left is None:
        best_dist = distances.dist(X_i, node_X, distance='squared_l2')
        return [node], [best_dist], 0

    # Internal node
    n_dimensions = X_i.shape[0]
    split_axis = _get_split_axis(depth, n_dimensions)

    # Recursively traverse down
    next_node = node.left if X_i[split_axis] < node_X[split_axis] else node.right
    best_nearest, best_dist, height = _find_nearest(X_i, next_node, k, depth + 1)

    if height <= 2:
        # Check siblings
        sibling_node = node.right if X_i[split_axis] < node_X[split_axis] else node.left
        sibling_nearest, sibling_dist, sibling_height = _find_nearest(X_i, sibling_node, k, depth + 1)

        # Current internal node
        current_node_dist = distances.dist(X_i, node_X, distance='squared_l2')

        # Compare the best candidate with the ones just found
        if best_dist > current_node_dist:
            best_dist = current_node_dist
            best_nearest = node

        if best_dist > sibling_dist:
            best_dist = sibling_dist
            best_nearest = sibling_nearest

    return best_nearest, best_dist, height + 1


def _get_split_axis(depth, n_dimensions):
    return depth % n_dimensions


class TreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        return "[val = {0}, left={1}, right={2}]".format(self.val, self.left, self.right)
