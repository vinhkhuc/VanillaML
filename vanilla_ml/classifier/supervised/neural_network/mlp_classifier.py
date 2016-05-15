"""
Multi-layer Perceptron (Feed-forward Neural Network)
"""
import numpy as np

from vanilla_ml.base.neural_network.containers import Sequential
from vanilla_ml.base.neural_network.layers import FeedForward
from vanilla_ml.base.neural_network.loss import CrossEntropyLoss
from vanilla_ml.classifier.supervised.abstract_classifier import AbstractClassifier


class MLPClassifier(AbstractClassifier):

    def __init__(self, layers, fit_bias=False, learning_rate=1.0,
                 mini_batch_size=10, max_iterations=50, tol=1e-5, verbose=True, random_state=42):

        assert learning_rate > 0, "Learning rate must be positive."

        # TODO: Support fit_bias
        assert not fit_bias, "fit_bias is not supported."

        self.layers = layers
        self.fit_bias = fit_bias
        self.lr = learning_rate
        self.mini_batch_size = mini_batch_size
        self.max_iterations = max_iterations
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self._classes = None
        self.model = None

    def fit(self, X, y, sample_weights=None):
        assert sample_weights is None, "Specifying sample weights is not supported!"
        assert len(X) == len(y), "Length mismatches: len(X) = %d, len(y) = %d" % (len(X), len(y))

        np.random.seed(self.random_state)

        # if self.fit_bias:
        #     X = np.hstack((X, np.ones((X.shape[0], 1))))

        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # # Convert y to one-hot matrix
        # one_hot_y = misc.one_hot(y, n_classes)

        # Model
        self.model = _build_model(self.layers, n_classes)

        # Cost
        loss = CrossEntropyLoss()
        loss.size_average = False
        loss.do_softmax_bprop = True

        # For report
        total_err  = 0.
        total_cost = 0.
        total_num  = 0

        # SGD params
        params = {
            "lrate": self.lr,
            "max_grad_norm": 40
        }

        # Run SGD
        indices = np.arange(n_samples)
        for it in range(self.max_iterations):
            if self.verbose and (it + 1) % 10 == 0:
                print("Iteration %d ..." % (it + 1))

            mini_batch = np.random.choice(indices, size=self.mini_batch_size, replace=False)
            input_data, target_data = X[mini_batch], y[mini_batch]

            # Forward propagation
            out = self.model.fprop(input_data)
            total_cost += loss.fprop(out, target_data)
            total_err  += loss.get_error(out, target_data)
            total_num  += self.mini_batch_size

            print("%d | train error: %g" % (total_num + 1, total_err / total_num))

            # Backward propagation
            grad = loss.bprop(out, target_data)
            self.model.bprop(input_data, grad)
            self.model.update(params)

    def predict_proba(self, X):
        return self.model.fprop(X)


def _build_model(layers, n_classes):

    model = Sequential()
    for layer in layers:
        model.add(FeedForward(layer[0], layer[1]))
    model.add(FeedForward(layers[-1][1], n_classes))
    model.modules[-1].skip_bprop = True

    return model
