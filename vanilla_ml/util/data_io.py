import gzip
import random
import csv
import zipfile
from os import path
import numpy as np

from vanilla_ml.util import misc
from vanilla_ml.util.misc import train_test_split, make_moons, make_blobs
from vanilla_ml.util.scaling.standard_scaler import StandardScaler

DOWNLOAD_DATASET_PATH = path.join(path.expanduser('~'), '.VanillaML', 'datasets')
MOVIE_LENS_100K_ZIP = 'ml-100k.zip'
MOVIE_LENS_100K_URL = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'

LOCAL_DATASET_PATH = path.join('..', '..', 'datasets')
IRIS_CSV_GZ = 'iris.csv.gz'
BOSTON_CSV_GZ = 'boston.csv.gz'
DIGITS_CSV_GZ = 'digits.csv.gz'


def _get_train_test_split(X, y, random_state=42):
    return train_test_split(X, y, test_size=0.25, random_state=random_state)


def load_iris():
    module_path = path.dirname(__file__)
    with gzip.open(path.join(module_path, LOCAL_DATASET_PATH, IRIS_CSV_GZ)) as f:
        reader = csv.reader(f)
        next(reader)  # skip the first row which contains meta-data
        data, targets = [], []
        for row in reader:
            data.append(np.array(row[:-1], np.float))
            targets.append(np.array(row[-1], np.int))
    return np.array(data), np.array(targets)


def load_digits():
    module_path = path.dirname(__file__)
    with gzip.open(path.join(module_path, LOCAL_DATASET_PATH, DIGITS_CSV_GZ)) as f:
        reader = csv.reader(f)
        data, targets = [], []
        for row in reader:
            data.append(np.array(row[:-1], np.float))
            targets.append(np.array(row[-1], np.int))
    return np.array(data), np.array(targets)


def load_boston():
    module_path = path.dirname(__file__)
    with gzip.open(path.join(module_path, LOCAL_DATASET_PATH, BOSTON_CSV_GZ)) as f:
        reader = csv.reader(f)
        next(reader)  # skip the first row which contains meta-data
        X, y = [], []
        for row in reader:
            X.append(np.array(row[:-1], np.float))
            y.append(np.array(row[-1], np.float))
    return np.array(X), np.array(y)


def get_xor_train_test(num_data_points=100):
    X = np.zeros((num_data_points, 2))
    y = np.zeros(num_data_points, dtype=int)
    for i in range(num_data_points):
        x = [random.randint(0, 1), random.randint(0, 1)]  # randomly generate 0, 1
        X[i] = x
        y[i] = x[0] ^ x[1]
    print("X = %s\ny = %s" % (X, y))

    return _get_train_test_split(X, y)


def get_iris_train_test():
    X, y = load_iris()
    return _get_train_test_split(X, y)


# Two linearly separated classes
def get_setosa_vericolour_iris_train_test():
    orig_X, orig_y = load_iris()

    # Setosa = 0, Vericolour = 1 (ref: https://archive.ics.uci.edu/ml/datasets/Iris)
    X = orig_X[orig_y != 2]
    y = orig_y[orig_y != 2]

    return _get_train_test_split(X, y)


# Two linearly non-separated classes
def get_vericolour_virginica_iris_train_test():
    orig_X, orig_y = load_iris()

    # Vericolour = 1, Virginica = 2 (ref: https://archive.ics.uci.edu/ml/datasets/Iris)
    X = orig_X[orig_y != 0]
    y = orig_y[orig_y != 0]
    y -= 1

    return _get_train_test_split(X, y)


def get_digits_train_test():
    X, y = load_digits()
    return _get_train_test_split(X, y)


# def get_20newsgroup_train_test(feature_selection=False):
#     # twenty_newsgroups = datasets.fetch_20newsgroups_vectorized(subset="test")
#     twenty_newsgroups = datasets.fetch_20newsgroups(subset="test")
#
#     vectorizer = CountVectorizer(dtype=np.int16)
#     X = vectorizer.fit_transform(twenty_newsgroups.data)
#     y = twenty_newsgroups.target
#
#     if feature_selection:
#         print("X's old shape = {0}".format(X.shape))
#         feature_selector = LinearSVC(C=1., penalty="l1", dual=False)
#         # feature_selector = SelectKBest(chi2, k=100)
#         print("Doing feature selection using {0} ...".format(feature_selector))
#         X_new = feature_selector.fit_transform(X, y)
#         X = X_new
#         print("X's new shape = {0}".format(X.shape))
#
#     return _get_train_test_split(X, y)


# def get_rcv1_train_test():
#     X, y = datasets.load_svmlight_file("../dataset/supervised/rcv1_train.multiclass")
#
#     # Filtering
#     X = X[y <= 2]
#     y = y[y <= 2]
#
#     return _get_train_test_split(X.toarray(), y)


def get_moons_train_test(num_samples=100):
    X, y = make_moons(n_samples=num_samples, noise=0.3, random_state=42)
    return _get_train_test_split(X, y)


def get_boston_train_test():
    X, y = load_boston()
    return _get_train_test_split(X, y)


def get_binary_classification_data(n_samples=750, centers=None, cluster_std=0.4, random_state=42):
    if centers is None:
        centers = np.array([[1, 1], [-1, -1]])

    assert len(centers) == 2, "There must be two classes for binary classification"
    return get_classification_data(n_samples, centers, cluster_std, random_state)


def get_classification_data(n_samples=750, centers=None, cluster_std=0.4, random_state=42):
    if centers is None:
        centers = [[1, 1], [-1, -1], [1, -1]]
    X, y = make_blobs(n_samples=n_samples, centers=centers,
                      cluster_std=cluster_std, random_state=random_state)
    X = StandardScaler().fit_transform(X)
    return X, y


def get_clustering_data(n_samples=750, centers=None, cluster_std=0.4, random_state=42):
    if centers is None:
        centers = [[1, 1], [-1, -1], [1, -1]]
    X, y = make_blobs(n_samples=n_samples, centers=centers,
                      cluster_std=cluster_std, random_state=random_state)
    X = StandardScaler().fit_transform(X)
    return X, y


def get_regression_line(noise=True):
    X = np.arange(0, 10, 0.1)
    y = 2 * X + 1
    X = X[:, None]
    if noise:
        y += np.random.normal(0, 1, size=len(X))
    return _get_train_test_split(X, y)


def get_regression_curve(noise=True):
    X = np.arange(0, 10, 0.1)
    y = X * np.sin(X)
    X = X[:, None]
    if noise:
        y += np.random.normal(0, 1, size=len(X))
    return _get_train_test_split(X, y)


# Adapted from http://qiita.com/sz_dr/items/0e50120318527a928407 (Japanese)
def get_ranking_train_test(n_dim=50, n_rank=5, n_sample=1000, sigma=5., random_state=42):
    rand = np.random
    rand.seed(random_state)
    y = rand.random_integers(n_rank, size=n_sample)
    w = rand.standard_normal(n_dim)
    X = [sigma * np.random.standard_normal(n_dim) + w * y_i for y_i in y]
    X = np.array(X, np.float32)
    return _get_train_test_split(X, y, random_state=random_state)


def get_movie_lens_100k_train_test(fold=None, random_state=42):
    """ Load training and test set from MovieLens 100k.

    Args:
        fold (Optional[int]): if None, the whole data set will be read,
         then will be randomly split into training and test sets.
        random_state (int):

    Returns:
        tuple: training and test sets

    """
    assert fold is None or 1 <= fold <= 5, "fold must be either from 1 to 5 or None."

    movie_lens_zipfile = path.join(DOWNLOAD_DATASET_PATH, MOVIE_LENS_100K_ZIP)
    if not path.exists(movie_lens_zipfile):
        print("Downloading the MovieLens data set '%s' ..." % movie_lens_zipfile)
        misc.download_file(MOVIE_LENS_100K_URL, movie_lens_zipfile)

    movie_lens = zipfile.ZipFile(movie_lens_zipfile, 'r')
    if fold is None:
        data_file = movie_lens.open('ml-100k/u.data')
        data = np.loadtxt(data_file)
        X, y = data[:, :2], data[:, 2]
        X, y = _get_train_test_split(X, y, random_state=random_state)
    else:
        train_file = movie_lens.open('ml-100k/u%d.base' % fold)
        test_file = movie_lens.open('ml-100k/u%d.test' % fold)
        X, y = np.loadtxt(train_file), np.loadtxt(test_file)

    # TODO: Load x, y to a matrix
    return X, y


def get_accuracy(model, train_test):
    tr_X, te_X, tr_y, te_y = train_test

    print("Fitting %s ..." % model.__class__.__name__)
    model.fit(tr_X, tr_y)

    print("Predicting ...")
    pred_y = model.run_qa(te_X)

    return (te_y == pred_y).mean()
