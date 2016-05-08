import random

import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.datasets.samples_generator import make_moons, make_blobs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing.data import StandardScaler
from sklearn.svm import LinearSVC


def _get_train_test_split(X, y):
    return train_test_split(X, y, test_size=0.25, random_state=10)


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
    iris = datasets.load_iris()
    return _get_train_test_split(iris.data, iris.target)


def get_digits_train_test():
    digits = datasets.load_digits()
    return _get_train_test_split(digits.data, digits.target)


def get_20newsgroup_train_test(feature_selection=False):
    # twenty_newsgroups = datasets.fetch_20newsgroups_vectorized(subset="test")
    twenty_newsgroups = datasets.fetch_20newsgroups(subset="test")

    vectorizer = CountVectorizer(dtype=np.int16)
    X = vectorizer.fit_transform(twenty_newsgroups.data)
    y = twenty_newsgroups.target

    if feature_selection:
        print("X's old shape = {0}".format(X.shape))
        feature_selector = LinearSVC(C=1., penalty="l1", dual=False)
            # SelectKBest(chi2, k=100)
        print("Doing feature selection using {0} ...".format(feature_selector))
        X_new = feature_selector.fit_transform(X, y)
        X = X_new
        print("X's new shape = {0}".format(X.shape))

    return _get_train_test_split(X, y)


def get_rcv1_train_test():
    X, y = datasets.load_svmlight_file("../dataset/supervised/rcv1_train.multiclass")

    # Filtering
    X = X[y <= 2]
    y = y[y <= 2]

    return _get_train_test_split(X.toarray(), y)


def get_moons_train_test(num_samples=100):
    X, y = make_moons(n_samples=num_samples, noise=0.3, random_state=42)
    return _get_train_test_split(X, y)


def get_boston_train_test():
    boston = datasets.load_boston()
    return _get_train_test_split(boston.data, boston.target)


def get_clustering_data(n_samples=750, centers=None, cluster_std=0.4, random_state=42):
    if centers is None:
        centers = [[1, 1], [-1, -1], [1, -1]]
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_state)
    X = StandardScaler().fit_transform(X)
    return X, y


def get_accuracy(model, train_test):
    tr_X, te_X, tr_y, te_y = train_test

    print("Fitting %s ..." % model.__class__.__name__)
    model.fit(tr_X, tr_y)

    print("Predicting ...")
    pred_y = model.run_qa(te_X)

    return (te_y == pred_y).mean()
