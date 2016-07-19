"""
Multi-layer Perceptron (Feed-forward Neural Network)
"""
import numpy as np

from vanilla_ml.base.neural_network.activators import Sigmoid, Softmax
from vanilla_ml.base.neural_network.containers import Sequential
from vanilla_ml.base.neural_network.layers import Linear
from vanilla_ml.base.neural_network.loss import CrossEntropyLoss
from vanilla_ml.supervised.classification.abstract_classifier import AbstractClassifier
from vanilla_ml.util.metrics.accuracy import accuracy_score


class MLPClassifier(AbstractClassifier):

    # TODO: n_epochs, tol shouldn't be in the constructor
    def __init__(self, layers, learning_rate=1.0, batch_size=10, n_epochs=50,
                 tol=1e-5, verbose=True, random_state=42):

        assert learning_rate > 0, "Learning rate must be positive."

        self.layers = layers
        self.lr = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self._classes = None
        self.input_size = None
        self.output_size = None
        self.model = None
        self.loss = None

    def fit(self, X, y, sample_weights=None):
        assert sample_weights is None, "Specifying sample weights is not supported!"
        assert len(X) == len(y), "Length mismatches: len(X) = %d, len(y) = %d" % (len(X), len(y))

        np.random.seed(self.random_state)

        n_samples, self.input_size = X.shape
        self._classes = np.unique(y)
        self.output_size = len(self._classes)

        # Model
        self.model, self.loss = self._build_model()

        # SGD params
        params = {"lrate": self.lr, "max_grad_norm": 40}

        indices = np.arange(n_samples)

        # Run SGD
        for epoch in range(self.n_epochs):
            if self.verbose and (epoch + 1) % 10 == 0:
                print("Epoch %d ..." % (epoch + 1))

            # For report
            total_err  = 0.
            total_cost = 0.
            total_num  = 0

            for it in range(n_samples / self.batch_size):

                # batch = np.random.choice(indices, size=self.batch_size, replace=False)
                start = it * self.batch_size
                end = min((it + 1) * self.batch_size, n_samples)
                batch = indices[start:end]
                input_data, target_data = X[batch], y[batch]

                # Forward propagation
                out = self.model.fprop(input_data)
                total_cost += self.loss.fprop(out, target_data)
                pred = out.argmax(axis=1)
                total_err += accuracy_score(pred, target_data)
                total_num += self.batch_size

                if self.verbose:
                    print("\n* Iter %d" % (it + 1))
                    print("loss = %s" % self.loss.fprop(out, target_data))
                    print("Accuracy = %.2f%%" % (100. * accuracy_score(target_data, pred)))

                # Backward propagation
                grad_output = self.loss.bprop(out, target_data)
                self.model.bprop(input_data, grad_output)
                self.model.update(params)

            if self.verbose:
                print("\n* Epoch %d" % (epoch + 1))
                print("%d | train error: %g" % (total_num + 1, total_err / total_num))
                print("pred =\n%s" % pred)
                print("target_data =\n%s" % target_data)
                print("accuracy = %.2f%%" % (100. * accuracy_score(pred, target_data)))

    def predict_proba(self, X):
        return self.model.fprop(X)

    def _build_model(self):
        input_size, layer_sizes, output_size = self.input_size, self.layers, self.output_size

        model = Sequential()
        for i in range(len(layer_sizes)):
            if i == 0:
                model.add(Linear(input_size, layer_sizes[i]))
            else:
                model.add(Linear(layer_sizes[i - 1], layer_sizes[i]))
            model.add(Sigmoid())
            # model.add(ReLU())

        model.add(Linear(layer_sizes[-1], output_size))
        model.add(Softmax(skip_bprop=True))

        # Cost
        loss = CrossEntropyLoss(size_average=True, do_softmax_bprop=True)

        return model, loss
