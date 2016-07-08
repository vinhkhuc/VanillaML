import unittest

from vanilla_ml.classifier.supervised.logistic_regression import LogisticRegression
from vanilla_ml.util import data_io
from vanilla_ml.util.metrics.accuracy import accuracy_score
from vanilla_ml.util.scaling.standard_scaler import StandardScaler


class TestLogisticRegression(unittest.TestCase):

    def test_iris_two_classes(self):
        # train_X, test_X, train_y, test_y = data_io.get_setosa_vericolour_iris_train_test()
        train_X, test_X, train_y, test_y = data_io.get_vericolour_virginica_iris_train_test()
        print("train_X's shape = %s, train_y's shape = %s" % (train_X.shape, train_y.shape))
        print("test_X's shape = %s, test_y's shape = %s" % (test_X.shape, test_y.shape))

        print("Applying standard scaling ...")
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

        clf = LogisticRegression(fit_bias=False, max_iterations=100, penalty_type='l2', penalty_factor=0.1)
        # clf = skLogisticRegression(penalty='l2')
        print("clf: %s" % clf)

        print("Fitting ...")
        clf.fit(train_X, train_y)

        print("Predicting ...")
        pred_y = clf.predict(test_X)
        print("y = %s" % test_y)
        print("pred_y = %s" % clf.predict(test_X))

        accuracy = accuracy_score(test_y, pred_y)
        print("Accuracy = %g%%" % (100 * accuracy))

        self.assertGreaterEqual(accuracy, 0.85)
