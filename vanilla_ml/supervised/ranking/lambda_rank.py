"""
LambdaRank

Ref: "From RankNet to LambdaRank to LambdaMART: An Overview", Christ Burges.
"""

from vanilla_ml.base.neural_network.activators import Sigmoid
from vanilla_ml.base.neural_network.containers import Sequential
from vanilla_ml.base.neural_network.layers import Linear
from vanilla_ml.base.neural_network.loss import LambdaRankLoss
from vanilla_ml.supervised.ranking.rank_net import RankNet


class LambdaRank(RankNet):
    def __init__(self, layers, learning_rate=1.0, batch_size=10,
                 n_epochs=50, tol=1e-5, verbose=True, random_state=42):
        super(LambdaRank, self).__init__(layers, learning_rate, batch_size,
                                         n_epochs, tol, verbose, random_state)

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
        loss = LambdaRankLoss(sigma=1, size_average=True)

        return model, loss
