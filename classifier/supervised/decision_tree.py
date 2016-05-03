"""
Decision tree works by recursively splitting the input space into regions
and create local model for each region.
"""
import numpy as np
from classifier.supervised.abstract_classifier import AbstractClassifier

class DecisionTreeClassifier(AbstractClassifier):

    def __init__(self, max_depth=3, criterion='entropy', verbose=False):
        super(AbstractClassifier, self).__init__()
        self.max_depth = max_depth
        self.criterion = criterion
        self.root = None  # root node
        self.classes_ = None  # a list of classes
        self.verbose = verbose

    def fit(self, X, y):
        assert len(X) == len(y), "Length mismatches: len(tr_X) = %d, len(tr_y) = %d" % (len(X), len(y))
        assert y.dtype == np.int, "y must be integers"
        assert np.all(y >= 0), "y must be non-negative"

        self.classes_ = np.unique(y)

        # Fit the tree starting from the root node
        root = TreeNode()
        self._fit_node(root, X, y, depth=0)
        self.root = root

    def _fit_node(self, node, X, y, depth):
        """
        Recursively fit this node and its left/right child nodes if there is a good split.
        """
        node.pred_prob = self._get_predicted_proba(y)

        # Check for stopping condition
        if not self.is_stopping(X, y, depth):
            # Find the best split
            j, t, l, r, min_cost_reduction = self._split(X, y)

            if _is_worth_splitting(X, y, j, t, l, r, min_cost_reduction):
                # Save split feature and split value. Also initialize left and right child nodes.
                node.split_feat, node.split_val, node.left, node.right = j, t, TreeNode(), TreeNode()
                self._fit_node(node.left, X[l], y[l], depth + 1)
                self._fit_node(node.right, X[r], y[r], depth + 1)

    def _split(self, X, y):
        """
        Find a split (based on the formula 16.5 from Kevin Murphy's book).
        """
        j_max = None  # the feature to split on
        t_max = None  # split value
        l_max = None  # indices of data points in the left child node
        r_max = None  # for the right child node
        delta_max = float("-inf")

        criterion = self.criterion
        num_features = X.shape[1]
        for j in range(num_features):
            X_j = X[:, j]
            feature_values = np.unique(X_j)
            for t in feature_values:
                l = np.where(X_j <= t)[0]
                r = np.where(X_j > t)[0]
                delta = self._cost(y, criterion) - (len(l) * self._cost(y[l], criterion)
                                                    + len(r) * self._cost(y[r], criterion)) / len(y)
                if delta_max < delta:
                    j_max, t_max, l_max, r_max, delta_max = j, t, l, r, delta
        return j_max, t_max, l_max, r_max, delta_max

    def _get_predicted_proba(self, y):
        # Class distribution
        y_prob = np.bincount(y) / float(len(y))

        # Fill zeros for classes that are not included in y
        diff = len(self.classes_) - len(y_prob)
        y_prob = np.pad(y_prob, (0, diff), mode='constant', constant_values=0)
        return y_prob

    def _cost(self, y, criterion):
        """
        Cost function
        """
        y_prob = self._get_predicted_proba(y)
        if criterion == 'entropy':
            log2_y_prob = np.log2(y_prob)
            log2_y_prob[log2_y_prob == -np.inf] = 0  # replace -infs by zeros since they'll be eliminated anyway.
            return -np.sum(y_prob * log2_y_prob)
        elif criterion == 'gini':
            return np.sum(y_prob * (1 - y_prob))
        else:
            raise Exception("Criterion must be either entropy or gini.")

    def predict_proba(self, X):
        y_pred = np.zeros((X.shape[0], len(self.classes_)))
        queue = [(self.root, np.arange(len(X)))]  # breadth-first traverse

        while len(queue) > 0:
            node, indices = queue.pop(0)

            # Check if this is a leaf node.
            # Non-leaf node has BOTH left AND right child nodes.
            if node.left is None and node.right is None:
                y_pred[indices] = node.pred_prob
            else:
                # Non-leaf node
                j, t = node.split_feat, node.split_val
                left_indices = indices[X[indices, j] <= t]
                right_indices = indices[X[indices, j] > t]

                if len(left_indices) > 0 and node.left is not None:
                    queue.append((node.left, left_indices))

                if len(right_indices) > 0 and node.right is not None:
                    queue.append((node.right, right_indices))
        return y_pred

    def is_stopping(self, X, y, depth):
        stopping = False
        if depth == self.max_depth:
            stopping = True
            if self.verbose:
                print("Max depth has been reached. Stop splitting further.")
        elif len(X) == 0:
            stopping = True
            if self.verbose:
                print("This node has no training data to make a split.")
        return stopping

    def print_tree(self):
        assert self.root is not None, "The tree has not been fitted!"
        print("ROOT: %s" % _get_node_structure(self.root, depth=0))


class TreeNode(object):
    pred_prob = None    # node's predicted probability
    split_feat = None   # split feature
    split_val = None    # split value
    left = None         # left child
    right = None        # right child


def _is_worth_splitting(X, y, j, t, l, r, min_cost_reduction):
    """
    Check the best split is worth considering.
    Always return True for now.
    """
    return True  # assume that the best split is always worth considering.


def _get_node_structure(node, depth):
    """ Recursively get tree structure.

    Args:
        node (TreeNode): current node.
        depth (int): current node's depth.

    Returns:
        str: tree structure starting from the given node.

    """
    s = "\t[%s]\tlabel=%d\n" % (", ".join(["%.2g" % prob for prob in node.pred_prob]), node.pred_prob.argmax())
    if node.left is not None:
        s += "|\t" * depth + "|--- feature[%d] <= %g: %s" % (node.split_feat, node.split_val,
                                                             _get_node_structure(node.left, depth + 1))
    if node.right is not None:
        s += "|\t" * depth + "|--- feature[%d] > %g: %s" % (node.split_feat, node.split_val,
                                                            _get_node_structure(node.right, depth + 1))
    return s
