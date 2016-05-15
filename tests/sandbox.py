import numpy as np
from sklearn.ensemble.forest import RandomForestClassifier as skRandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier as skGradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree.tree import DecisionTreeClassifier as skDecisionTreeClassifier

from vanilla_ml.util import data_io
from vanilla_ml.classifier.supervised.adaboost_classifier import AdaBoostClassifier
from vanilla_ml.classifier.supervised.decision_tree_classifier import DecisionTreeClassifier
from vanilla_ml.classifier.supervised.naive_bayes import NaiveBayesClassifier


def test_sklearn():
    # train_test = get_digits_train_test()
    train_test = data_io.get_20newsgroup_train_test()
    # train_test = get_rcv1_train_test()

    models = [
        # MultinomialNB(fit_prior=False),
        BernoulliNB(fit_prior=False),
        # LogisticRegression(),
        # RandomForestClassifier(),
        # GradientBoostingClassifier()
    ]

    for model in models:
        accuracy = data_io.get_accuracy(model, train_test)
        print("* Model = %r\n  Accuracy = %f\n" % (model, accuracy))


def test_my_naive_bayes():
    train_test = data_io.get_digits_train_test()
    # train_test = get_20newsgroup_train_test()
    # train_test = get_rcv1_train_test()

    tr_X, te_X, tr_y, te_y = train_test

    # Filtering
    # tr_X = tr_X[tr_y <= 2]
    # tr_y = tr_y[tr_y <= 2]

    nb = NaiveBayesClassifier()
    print("Fitting Naive Bayes ...")
    nb.fit(tr_X, tr_y)

    print("Predicting ...")
    pred_y = nb.predict(te_X)
    print(te_y == pred_y).mean()



def test_adaboost():
    train_X, test_X, train_y, test_y = data_io.get_moons_train_test()

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


def test_random_forest():
    # train_X, test_X, train_y, test_y = data_io.get_xor_train_test(50)
    # train_X, test_X, train_y, test_y = data_io.get_iris_train_test()
    train_X, test_X, train_y, test_y = data_io.get_moons_train_test(200)
    print("train_X's shape = %s, train_y's shape = %s" % (train_X.shape, train_y.shape))
    print("test_X's shape = %s, test_y's shape = %s" % (test_X.shape, test_y.shape))

    # clf = RandomForestClassifier(num_trees=20, max_depth=3, criterion='entropy', rand_state=42)
    clf = skRandomForestClassifier(n_estimators=20, max_depth=3, criterion='entropy', random_state=42)

    print("Fitting ...")
    clf.fit(train_X, train_y)

    print("Predicting ...")
    y_pred = clf.predict(test_X)

    print("Predictions = %s" % y_pred)
    print("Correct = %s" % test_y)
    print("Accuracy = %g%%" % (100. * np.mean(y_pred == test_y)))


def test_gradient_boosting():
    train_X, test_X, train_y, test_y = data_io.get_moons_train_test()

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
    # test_adaboost()
    # test_random_forest()
    # test_gradient_boosting()

    """
    A simple example of an animated plot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots()

    x = np.arange(0, 2*np.pi, 0.01)
    line, = ax.plot(x, np.sin(x))

    def animate():
        # line.set_ydata(np.sin(x + i/10.0))  # update the data
        # return line,
        # print(i)
        pass

    # # Init only required for blitting to give a clean slate.
    # def init():
    #     line.set_ydata(np.ma.array(x, mask=True))
    #     return line,

    # ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
    #                               interval=25, blit=True)
    ani = animation.FuncAnimation(fig, animate, interval=25, blit=False)
    plt.show()
    print("Done")
