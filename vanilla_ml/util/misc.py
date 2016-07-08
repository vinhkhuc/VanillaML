"""
Misc utility
"""
import numpy as np
from __future__ import division


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def sign(x):
    return 1 if x >= 0 else -1


def get_penalty(w, factor, penalty):
    """ Get penalty for the input ndarray.

    Args:
        w (ndarray): input data, shape K x P
        factor (float): penalty factor
        penalty (str): penalty type

    Returns:
        ndarray: penalty values

    """
    assert factor > 0, "The penalty factor must be positive."

    if penalty == 'l1':
        raise Exception("L1 penalty is not supported yet!")
    elif penalty == 'l2':
        return 2 * factor * w
    else:
        raise Exception("The penalty '%s' is not supported!" % penalty)


def sign_prediction(y):
    """
    Maps {0, 1} to {-1, 1}.
    """
    return 2 * y - 1


def unsign_prediction(y):
    """
    Maps {-1, 1} to {0, 1}.
    """
    return (y + 1) / 2


def one_hot(y, n_classes=None):
    """ Convert an 1D array to one-hot 2D array (using one-liner trick).

    Args:
        y (ndarray): uint array, shape N
        n_classes (Optional[int]): number of classes.

    Returns:
        ndarray: one-hot 2D array, shape N x K

    """
    if n_classes is None:
        n_classes = len(np.unique(y))
    return np.eye(n_classes)[y]


def softmax(X):
    """ Compute softmax (based on the formulas 3.70 and 8.33 in Kevin Murphy's book).

    Args:
        X (ndarray): array, shape N x K

    Returns:
        ndarray: softmax, shape N x 1.

    """
    log_sum_exp_X = log_sum_exp(X)
    return np.exp(X - log_sum_exp_X[:, None])


def log_sum_exp(X):
    """ Compute log of sum of exps.
    Using the log-sum-exp trick as shown in the formula 3.74 in Kevin Murphy's book.

    Args:
        X (ndarray): array, shape N x K

    Returns:
        ndarray: log-sum-exp results, shape N x 1.

    """
    max_X = X.max(axis=1)
    return max_X + np.log(np.sum(np.exp(X - max_X[:, None]), axis=1))


def train_test_split(X, y, test_size=0.25, random_state=42):
    """ Split the data set into training and test sets.

    Args:
        X (ndarray): data
        y (ndarray): target
        test_size (float): percentage of the test set
        random_state (int): random state

    Returns:
        tuple: a tuple of X_train, X_test, y_train, y_test

    """
    assert X.shape[0] == y.shape[0], "X, y have mismatched lengths"
    orig_size = X.shape[0]
    train_size = int(orig_size * (1 - test_size))
    np.random.seed(random_state)
    rand_indices = np.random.permutation(orig_size)
    train_indices, test_indices = \
        rand_indices[:train_size], rand_indices[train_size:]

    return X[train_indices], X[test_indices], \
           y[train_indices], y[test_indices]


# Adapted from sklearn's make_moons()
def make_moons(n_samples, noise=0.3, random_state=42, factor=.8):
    """ Generate two moons.

    Args:
        n_samples (int): number of samples to generate
        noise (float): noise level
        random_state (int): random state

    Returns:
        tuple: a tuple of X and y

    """
    linspace = np.linspace(0, 2 * np.pi, n_samples // 2 + 1)[:-1]
    outer_circ_x = np.cos(linspace)
    outer_circ_y = np.sin(linspace)
    inner_circ_x = outer_circ_x * factor
    inner_circ_y = outer_circ_y * factor

    X = np.vstack((np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y))).T
    y = np.hstack([np.zeros(n_samples // 2, dtype=np.intp),
                   np.ones(n_samples // 2, dtype=np.intp)])

    # Add noise
    X += np.random.normal(scale=noise, size=X.shape)

    # Shuffle
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(n_samples)
    return X[shuffled_indices], y[shuffled_indices]


# Adapted from sklearn's make_blobs
def make_blobs(n_samples=100, n_features=2, n_centers=3, centers=None, cluster_std=1.0,
               center_range=(-10.0, 10.0), random_state=None):
    """ Draw random data from Gaussian for clustering.

    Args:
        n_samples (int): number of samples to generate.
        n_features (int): number of features, i.e. data point's dimensions.
        n_centers (int): number of clusters
        centers (Optional[ndarray]): an array of shape [n_centers, n_features]
                                    for center locations.
        cluster_std (float): cluster's standard deviation.
        center_range (tuple): a tuple of min and max values of cluster centers.
        random_state (int): random state.

    Returns:
        tuple: a tuple of data and target.

    """
    np.random.seed(random_state)

    # Generate random cluster centers if they are not given
    if centers is None:
        center_low, center_high = center_range
        centers = np.random.uniform(center_low, center_high,
                                    size=(centers, n_features))
    else:
        n_features = centers.shape[1]

    # Calculate number of samples per center
    n_samples_per_center = [int(n_samples // centers)] * centers
    for i in range(n_samples % centers):
        n_samples_per_center[i] += 1

    # Generate blobs
    X = []
    y = []
    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        X.append(centers[i] + np.random.normal(scale=std, size=(n, n_features)))
        y += [i] * n

    X = np.array(X)
    y = np.array(y)
    rand_indices = np.random.permutation(n_samples)
    return X[rand_indices], y[rand_indices]
