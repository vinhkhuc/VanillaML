"""
Decision tree works by recursively splitting the input space into regions and create local model for each region.
"""
import numpy as np


class DecisionTreeBase(object):
    """
    Base class for Decision Tree Classifier/Regressor.
    """
    ALLOWED_CRITERIA = {'gini', 'entropy', 'mse'}

    def __init__(self,
                 max_depth=3,
                 criterion=None,
                 min_leaf_samples=1,
                 rand_features_ratio=None,
                 rand_state=42,
                 verbose=False):

        assert criterion in DecisionTreeBase.ALLOWED_CRITERIA, \
            "The split criterion '%s' is not supported." % criterion

        assert min_leaf_samples > 0, \
            "Minimum number of samples in leaf nodes must be positive."

        assert rand_features_ratio is None or 0 < rand_features_ratio <= 1, \
            "Ratio of random features must be in (0, 1]."

        self.max_depth = max_depth
        self.criterion = criterion
        self.min_leaf_samples = min_leaf_samples
        self.rand_features_ratio = rand_features_ratio
        self.verbose = verbose
        self.root = None  # root node

        self.is_classifier = criterion == 'gini' or criterion == 'entropy'
        if self.is_classifier:
            self._classes = None

        # Set random state if random features will be used (i.e. for random forest).
        if rand_features_ratio is not None:
            np.random.seed(rand_state)

    def fit(self, X, y, sample_weights=None):

        assert len(X) == len(y), "Length mismatches: len(X) = %d, len(y) = %d" % (len(X), len(y))
        assert np.all(y >= 0), "y must be non-negative"

        if self.is_classifier:
            self._classes = np.unique(y)

        y = y.astype(int)
        if sample_weights is None:
            sample_weights = np.ones(len(y)) / float(len(y))

        # Fit the tree starting from the root node
        root = TreeNode()
        self._fit_node(root, X, y, sample_weights, depth=0)
        self.root = root

    def _fit_node(self, node, X, y, w, depth):
        """
        Recursively fit this node and its left/right child nodes if there is a good split.
        """
        if self.is_classifier:
            node.pred = self._get_weighted_proba(y, w)
        else:
            node.pred = self._get_weighted_average(y, w)

        # Check for stopping condition
        if not self._is_stopping(X, y, depth):
            # Find the best split
            j, t, l, r, min_cost_reduction = self._split(X, y, w)

            if self._is_worth_splitting(X, y, j, t, l, r, min_cost_reduction):
                # Save split feature and split value. Also initialize left and right child nodes.
                node.split_feat, node.split_val, node.left, node.right = j, t, TreeNode(), TreeNode()
                self._fit_node(node.left, X[l], y[l], w[l], depth + 1)
                self._fit_node(node.right, X[r], y[r], w[r], depth + 1)

    def _split(self, X, y, w):
        """
        Find a split (based on the formula 16.5 from Kevin Murphy's book).
        """
        j_max = None  # the feature to split on
        t_max = None  # split value
        l_max = None  # indices of data points in the left child node
        r_max = None  # for the right child node
        delta_max = float("-inf")
        criterion = self.criterion

        # Select a random subset of features if ratio_rand_features is specified.
        total_features = X.shape[1]
        if self.rand_features_ratio is None:
            features = range(total_features)
        else:
            num_rand_features = int(total_features * self.rand_features_ratio)
            features = np.random.choice(total_features, size=num_rand_features, replace=False)
            features.sort()

        # Find the best split
        for j in features:
            X_j = X[:, j]
            feature_values = np.unique(X_j)
            for t in feature_values:
                l = np.where(X_j <= t)[0]
                r = np.where(X_j > t)[0]
                delta = self._cost(y, w, criterion) - (len(l) * self._cost(y[l], w[l], criterion)
                                                       + len(r) * self._cost(y[r], w[r], criterion)) / len(y)
                if delta_max < delta:
                    j_max, t_max, l_max, r_max, delta_max = j, t, l, r, delta

        return j_max, t_max, l_max, r_max, delta_max

    def _cost(self, y, w, criterion):
        """ Cost function

        Args:
            y (ndarray): sample classes N x 1
            w (ndarray): sample weights N x 1
            criterion (str): split criterion

        Returns:
            float: cost corresponding to the given criterion.

        """
        if self.is_classifier:
            y_prob = self._get_weighted_proba(y, w)
            if criterion == 'gini':
                return np.sum(y_prob * (1 - y_prob))
            else:
                log2_y_prob = np.log2(y_prob)
                # Replace -infs by zeros since they'll be eliminated anyway.
                log2_y_prob[log2_y_prob == -np.inf] = 0
                return -np.sum(y_prob * log2_y_prob)
        else:
            if len(y) == 0:
                return 0
            else:
                weighted_mean = np.average(y, weights=w)
                return np.mean(np.square(y - weighted_mean))

    def _get_weighted_proba(self, y, w):
        """ Get weighted prediction probability.

        Args:
            y (ndarray): sample classes, shape N x 1.
            w (ndarray): sample weights, shape N x 1.

        Returns:
            ndarray: weighted prediction probability.

        """
        # Class distribution
        y_weighted_count = np.bincount(y, w)
        y_prob = y_weighted_count / sum(y_weighted_count)

        # Fill zeros for classes that are not included in y
        diff = len(self._classes) - len(y_prob)
        y_prob = np.pad(y_prob, (0, diff), mode='constant', constant_values=0)
        return y_prob

    @staticmethod
    def _get_weighted_average(y, w):
        """ Get weighted average.

        Args:
            y (ndarray): sample values, shape N x 1.
            w (ndarray): sample weights, shape N x 1.

        Returns:
            ndarray: weighted average.

        """
        return np.average(y, weights=w)

    def predict(self, X):

        if self.is_classifier:
            y_pred = np.zeros((X.shape[0], len(self._classes)))
        else:
            y_pred = np.zeros(X.shape[0])

        queue = [(self.root, np.arange(len(X)))]  # breadth-first traversal
        while len(queue) > 0:
            node, indices = queue.pop(0)

            # Check if this is a leaf node.
            # Non-leaf node has BOTH left AND right child nodes.
            if node.left is None and node.right is None:
                y_pred[indices] = node.pred
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

    def _is_stopping(self, X, y, depth):
        stopping = False
        if depth == self.max_depth:
            stopping = True
            if self.verbose:
                print("Max depth has been reached. Stop splitting further.")
        elif len(X) < self.min_leaf_samples:
            stopping = True
            if self.verbose:
                print("This node has no training data to make a split.")
        return stopping

    def _is_worth_splitting(self, X, y, j, t, l, r, min_cost_reduction):
        """
        Check the best split is worth considering.
        """
        worth_splitting = True
        if len(l) < self.min_leaf_samples or len(r) < self.min_leaf_samples:
            worth_splitting = False
        return worth_splitting

    def print_tree(self):
        assert self.root is not None, "The tree has not been fitted!"
        print("ROOT: %s" % self._get_node_structure(self.root, depth=0))

    def _get_node_structure(self, node, depth):
        """ Recursively get tree structure.

        Args:
            node (TreeNode): current node.
            depth (int): current node's depth.

        Returns:
            str: tree structure starting from the given node.

        """
        if self.is_classifier:
            s = "\t[%s]\tlabel=%d\n" % (", ".join(["%.2g" % val for val in node.pred]), node.pred.argmax())
        else:
            s = "\t[%g]\n" % node.pred

        if node.left is not None:
            s += "|\t" * depth + "|--- feature[%d] <= %g: %s" % (node.split_feat, node.split_val,
                                                                 self._get_node_structure(node.left, depth + 1))
        if node.right is not None:
            s += "|\t" * depth + "|--- feature[%d] > %g: %s" % (node.split_feat, node.split_val,
                                                                self._get_node_structure(node.right, depth + 1))
        return s


class TreeNode(object):
    pred = None         # node's prediction
    split_feat = None   # split feature
    split_val = None    # split value
    left = None         # left child
    right = None        # right child
