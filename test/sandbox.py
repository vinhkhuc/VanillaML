import random
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.datasets.samples_generator import make_moons
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.tree.tree import DecisionTreeClassifier as skDecisionTreeClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier as skAdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier as skGradientBoostingClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier as skGradientBoostingRegressor

from classifier.supervised.adaboost import AdaBoostClassifier
from classifier.supervised.decision_tree import DecisionTreeClassifier
from classifier.supervised.naive_bayes import NaiveBayes


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


def get_moons_train_test():
    X, y = make_moons(noise=0.3, random_state=0)
    return _get_train_test_split(X, y)


def get_accuracy(model, train_test):
    tr_X, te_X, tr_y, te_y = train_test

    print("Fitting %s ..." % model.__class__.__name__)
    model.fit(tr_X, tr_y)

    print("Predicting ...")
    pred_y = model.run_qa(te_X)

    return (te_y == pred_y).mean()


def test_sklearn():
    # train_test = get_digits_train_test()
    train_test = get_20newsgroup_train_test()
    # train_test = get_rcv1_train_test()

    models = [
        # MultinomialNB(fit_prior=False),
        BernoulliNB(fit_prior=False),
        # LogisticRegression(),
        # RandomForestClassifier(),
        # GradientBoostingClassifier()
    ]

    for model in models:
        accuracy = get_accuracy(model, train_test)
        print("* Model = %r\n  Accuracy = %f\n" % (model, accuracy))


def test_my_naive_bayes():
    train_test = get_digits_train_test()
    # train_test = get_20newsgroup_train_test()
    # train_test = get_rcv1_train_test()

    tr_X, te_X, tr_y, te_y = train_test

    # Filtering
    # tr_X = tr_X[tr_y <= 2]
    # tr_y = tr_y[tr_y <= 2]

    nb = NaiveBayes()
    print("Fitting Naive Bayes ...")
    nb.fit(tr_X, tr_y)

    print("Predicting ...")
    pred_y = nb.predict(te_X)
    print(te_y == pred_y).mean()


def test_decision_tree():
    # train_X, test_X, train_y, test_y = get_xor_train_test(50)
    # train_X, test_X, train_y, test_y = get_iris_train_test()
    train_X, test_X, train_y, test_y = get_moons_train_test()
    print("train_X's shape = %s, train_y's shape = %s" % (train_X.shape, train_y.shape))
    print("test_X's shape = %s, test_y's shape = %s" % (test_X.shape, test_y.shape))

    # clf = DecisionTreeClassifier(max_depth=1, criterion='entropy')
    clf = skDecisionTreeClassifier(max_depth=1, criterion='entropy')
    print("clf: %s" % clf)

    print("Fitting ...")
    clf.fit(train_X, train_y)

    print("Predicting ...")
    pred_y = clf.predict(test_X)
    print("Accuracy = %g%%" % (100. * (test_y == pred_y).mean()))

    # print("Tree structure\n")
    # clf.print_tree()

    # print("Predicted: %s" % pred_y)
    #
    # prob_y = clf.predict_proba(test_X)
    # print("prob_y: %s" % prob_y)

    # for py, ty in zip(pred_y, test_y):
    #     if py != ty:
    #         print("Expected %d, got %d" % (py, ty))


def test_adaboost():
    train_X, test_X, train_y, test_y = get_moons_train_test()

    base_clf = DecisionTreeClassifier(max_depth=1, criterion="entropy")  # decision stump
    # base_clf = skDecisionTreeClassifier(max_depth=1, criterion="entropy")
    #
    clf = AdaBoostClassifier(base_clf, num_rounds=50)
    # clf = skAdaBoostClassifier(base_clf, n_estimators=50, algorithm='SAMME')
    # clf = base_clf

    print("Fitting ...")
    clf.fit(train_X, train_y)

    print("Predicting ...")
    y_pred = clf.predict(test_X)

    print("Predictions = %s" % y_pred)
    print("Correct = %s" % test_y)
    print("Accuracy = %g%%" % (100. * np.mean(y_pred == test_y)))


def test_gradient_boosting():
    train_X, test_X, train_y, test_y = get_moons_train_test()

    # clf = GradientBoostingClassifier(base_clf, num_rounds=50)
    clf = skGradientBoostingClassifier(n_estimators=50)
    # clf = DecisionTreeClassifier(max_depth=10)

    print("Fitting ...")
    clf.fit(train_X, train_y)

    print("Predicting ...")
    y_pred = clf.predict(test_X)

    print("Predictions = %s" % y_pred)
    print("Correct = %s" % test_y)
    print("Accuracy = %g%%" % (100. * np.mean(y_pred == test_y)))


if __name__ == "__main__":
    # test_sklearn()
    # test_my_naive_bayes()
    # test_decision_tree()
    # test_adaboost()
    test_gradient_boosting()
    print("Done")
