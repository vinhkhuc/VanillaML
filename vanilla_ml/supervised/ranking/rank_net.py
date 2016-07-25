"""
RankNet using Feed-forward Neural Network.

1) "From RankNet to LambdaRank to LambdaMART: An Overview", Christ Burges.
2) "Learning to Rank using Gradient Descent", Chris Burges et. al.
"""
import numpy as np

from vanilla_ml.base.neural_network.activators import Sigmoid
from vanilla_ml.base.neural_network.containers import Sequential
from vanilla_ml.base.neural_network.layers import Linear
from vanilla_ml.base.neural_network.loss import RankNetLoss
from vanilla_ml.supervised.ranking.abstract_ranker import AbstractRanker
from vanilla_ml.util.metrics.ranking import ndcg


class RankNet(AbstractRanker):

    def __init__(self, layers, learning_rate=1.0, batch_size=10,
                 n_epochs=50, tol=1e-5, verbose=True, random_state=42):
        assert learning_rate > 0, "Learning rate must be positive."

        self.layers = layers
        self.lr = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.input_size = None
        self.model = None
        self.loss = None

    def fit(self, X, y, sample_weights=None):
        assert sample_weights is None, "Specifying sample weights is not supported!"
        assert len(X) == len(y), "Length mismatches: len(X) = %d, len(y) = %d" % (len(X), len(y))

        np.random.seed(self.random_state)
        n_samples, self.input_size = X.shape

        # Model
        self.model, self.loss = self._build_model()

        # SGD params
        params = {"lrate": self.lr, "max_grad_norm": 40}
        indices = np.arange(n_samples)

        # Run SGD
        for epoch in range(self.n_epochs):
            if self.verbose:
                print("\n*Epoch %d:" % (epoch + 1))

            # For report
            total_num  = 0
            total_cost = 0.
            total_ndcg_score  = 0.

            for it in range(n_samples / self.batch_size):

                # batch = np.random.choice(indices, size=self.batch_size, replace=False)
                start = it * self.batch_size
                end = min((it + 1) * self.batch_size, n_samples)
                batch = indices[start:end]
                input_data, target_data = X[batch], y[batch]

                # Forward propagation
                pred = self.model.fprop(input_data)
                cost = self.loss.fprop(pred, target_data)
                total_num  += len(batch)
                total_cost += cost
                ndcg_score = ndcg(target_data, pred, k=5)
                total_ndcg_score += ndcg_score

                if self.verbose:
                    print("Iter %d, cost: %g, train NDCG@10: %g" %
                          (it + 1, ndcg_score, cost))

                # Backward propagation
                grad_output = self.loss.bprop(pred, target_data)
                self.model.bprop(input_data, grad_output)
                self.model.update(params)

            if self.verbose:
                print("Total: %d, avg train cost: %g, avg NDCG@10: %g" %
                      (total_num, total_cost / total_num, total_ndcg_score / total_num))

    def rank_score(self, X):
        return self.model.fprop(X).ravel()

    def _build_model(self):
        input_size, layer_sizes = self.input_size, self.layers
        model = Sequential()
        for i in range(len(layer_sizes)):
            if i == 0:
                model.add(Linear(input_size, layer_sizes[i]))
            else:
                model.add(Linear(layer_sizes[i - 1], layer_sizes[i]))
            model.add(Sigmoid())
            # model.add(ReLU())

        model.add(Linear(layer_sizes[-1], 1))

        # Cost
        loss = RankNetLoss(sigma=1, size_average=True)

        return model, loss
