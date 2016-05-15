import unittest

from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.preprocessing.data import StandardScaler

from vanilla_ml.classifier.supervised.linear_svm import LinearSVM
from vanilla_ml.util import data_io
from vanilla_ml.util.metrics.accuracy import accuracy_score


class TestLinearSVM(unittest.TestCase):

    def test_iris_two_classes(self):
        train_X, test_X, train_y, test_y = data_io.get_vericolour_virginica_iris_train_test()
        print("train_X's shape = %s, train_y's shape = %s" % (train_X.shape, train_y.shape))
        print("test_X's shape = %s, test_y's shape = %s" % (test_X.shape, test_y.shape))

        print("Applying standard scaling ...")
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

        clf = LinearSVM(max_iterations=2)
        # clf = SGDClassifier(loss='hinge', penalty='l2', n_iter=100)
        print("clf: %s" % clf)

        print("Fitting ...")
        clf.fit(train_X, train_y)

        print("Predicting ...")
        pred_y = clf.predict(test_X)
        print("y = %s" % test_y)
        print("pred_y = %s" % clf.predict(test_X))

        accuracy = accuracy_score(test_y, pred_y)
        print("Accuracy = %g%%" % (100 * accuracy))

        # self.assertGreaterEqual(accuracy, 0.95)

if __name__ == '__main__':
    unittest.main()