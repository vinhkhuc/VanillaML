import unittest

from sklearn.preprocessing.data import StandardScaler

from vanilla_ml.supervised.classification.mlp_classifier import MLPClassifier
from vanilla_ml.util import data_io
from vanilla_ml.util.metrics.accuracy import accuracy_score


class TestMLPClassifier(unittest.TestCase):

    def test_iris(self):
        train_X, test_X, train_y, test_y = data_io.get_iris_train_test()
        print("train_X's shape = %s, train_y's shape = %s" % (train_X.shape, train_y.shape))
        print("test_X's shape = %s, test_y's shape = %s" % (test_X.shape, test_y.shape))

        print("Applying standard scaling ...")
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

        # train_X = test_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        # train_y = test_y = np.array([0, 1, 1, 0])

        # train_X = test_X = np.array([[0], [1]])
        # train_y = test_y = np.array([0, 1])

        layers = [10]
        clf = MLPClassifier(layers, batch_size=train_X.shape[0], n_epochs=100, learning_rate=0.1)
        print("clf: %s" % clf)

        print("Fitting ...")
        clf.fit(train_X, train_y)

        print("Predicting ...")
        pred_y = clf.predict(test_X)
        print("y = %s" % test_y)
        print("pred_y = \n%s" % pred_y)

        # pred_proba_y = clf.predict_proba(test_X)
        # print("pred_proba_y = \n%s" % pred_proba_y)

        accuracy = accuracy_score(test_y, pred_y)
        print("Accuracy = %g%%" % (100 * accuracy))

        # self.assertGreaterEqual(accuracy, 0.94)
