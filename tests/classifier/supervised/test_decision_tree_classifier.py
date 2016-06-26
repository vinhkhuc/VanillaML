import unittest

from vanilla_ml.classifier.supervised.decision_tree_classifier import DecisionTreeClassifier
from vanilla_ml.util import data_io


class TestDecisionTreeClassifier(unittest.TestCase):

    def test_two_moons(self):
        train_X, test_X, train_y, test_y = data_io.get_moons_train_test(num_samples=200)
        print("train_X's shape = %s, train_y's shape = %s" % (train_X.shape, train_y.shape))
        print("test_X's shape = %s, test_y's shape = %s" % (test_X.shape, test_y.shape))

        clf = DecisionTreeClassifier(max_depth=1, criterion='entropy')
        # clf = skDecisionTreeClassifier(max_depth=1, criterion='entropy', min_samples_split=1, min_samples_leaf=1)
        print("clf: %s" % clf)

        print("Fitting ...")
        clf.fit(train_X, train_y)

        print("Predicting ...")
        pred_y = clf.predict(test_X)
        print("prob_y = %s" % clf.predict_proba(test_X))

        accuracy = (test_y == pred_y).mean()
        print("Accuracy = %g%%" % (100 * accuracy))

        self.assertGreaterEqual(accuracy, 0.7)
