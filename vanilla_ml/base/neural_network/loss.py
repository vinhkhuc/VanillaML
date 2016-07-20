import itertools
from abc import ABCMeta, abstractmethod

import numpy as np


class Loss(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fprop(self, input_data, target_data):
        pass

    @abstractmethod
    def bprop(self, input_data, target_data):
        pass


class MSELoss(Loss):

    def __init__(self, size_average=True):
        self.size_average = size_average

    def fprop(self, input_data, target_data):
        cost = np.sum(np.square(target_data - input_data))
        if self.size_average:
            cost /= target_data.shape[0]
        return cost

    def bprop(self, input_data, target_data):
        grad_input = 2 * (input_data - target_data)
        if self.size_average:
            grad_input /= target_data.shape[0]
        return grad_input


class CrossEntropyLoss(Loss):

    def __init__(self, eps=1e-7, size_average=True, do_softmax_bprop=False):
        self.eps = eps
        self.size_average = size_average
        self.do_softmax_bprop = do_softmax_bprop

    def fprop(self, input_data, target_data):
        # tmp = [(t, i) for i, t in enumerate(target_data)]
        # z = zip(*tmp)  # unzipping trick !

        tmp = [(i, t) for i, t in enumerate(target_data)]
        z = zip(*tmp)  # unzipping trick !

        # z = target_data
        cost = -np.sum(np.log(input_data[z]))
        if self.size_average:
            cost /= input_data.shape[0]

        return cost

    def bprop(self, input_data, target_data):
        # tmp = [(t, i) for i, t in enumerate(target_data)]
        # z = zip(*tmp)

        tmp = [(i, t) for i, t in enumerate(target_data)]
        z = zip(*tmp)  # unzipping trick !

        # z = target_data

        if self.do_softmax_bprop:
            grad_input = np.copy(input_data)
            grad_input[z] -= 1
        else:
            grad_input = np.zeros_like(input_data, np.float32)
            grad_input[z] = -1. / (input_data[z] + self.eps)

        if self.size_average:
            grad_input /= input_data.shape[0]

        return grad_input


class RankNetLoss(Loss):
    """
    Loss for RankNet
    See the section 2 about RankNet in
        "From RankNet to LambdaRank to LambdaMART: An Overview", Christ Burges.
    """
    def __init__(self, sigma=1., size_average=True):
        self.sigma = sigma
        self.size_average = size_average

    def fprop(self, input_data, target_data):
        # Try to use the same notation as in the paper
        s, y, sigma = input_data, target_data, self.sigma

        # Iterate over all combinations of indices, i.e (0, 0), (0, 1), ...
        n_samples = s.shape[0]
        cost = 0
        # TODO: This loop makes the computation slow
        for i, j in itertools.combinations(range(n_samples), 2):
            s_ij = s[i] - s[j]
            S_ij = 1 if y[i] > y[j] else -1 if y[i] < y[j] else 0
            cost += 0.5 * sigma * (1 - S_ij) * s_ij + np.log(1 + np.exp(-sigma * s_ij))

        # ij = [(i, j) for i, j in itertools.combinations(range(n_samples), 2)]
        # s_ij = [s[i] - s[j] for i, j in ij]
        # S_ij = [1 if y[i] > y[j] else -1 if y[i] < y[j] else 0 for i, j in ij]
        # s_ij, S_ij = np.array(s_ij), np.array(S_ij)
        #
        # cost = np.sum(0.5 * sigma * (1 - S_ij) * s_ij + np.log(1 + np.exp(-sigma * s_ij)))

        if self.size_average:
            cost /= 0.5 * n_samples * (n_samples + 1)  # normalized by the total number of pairs
        return cost

    def bprop(self, input_data, target_data):
        """ Back-propagation. Here we use the approach of calculating gradient
            as shown in the section 2.1 in the paper.
        """
        s, y, sigma = input_data, target_data, self.sigma
        n_samples = s.shape[0]

        grad_input = np.zeros_like(input_data, np.float32)  # grad_input is lambda (as in the paper)
        # TODO: This loop makes the computation slow
        for i, j in itertools.combinations(range(n_samples), 2):
            S_ij = 1 if y[i] > y[j] else -1 if y[i] < y[j] else 0
            s_ij = s[i] - s[j]
            lambda_ij = 0.5 * sigma * (1 - S_ij) - 1 / (1 + np.exp(sigma * s_ij))  # dcost/ds_i
            grad_input[i] += lambda_ij
            grad_input[j] -= lambda_ij

        # ij = [(i, j) for i, j in itertools.combinations(range(n_samples), 2)]
        # s_ij = [s[i] - s[j] for i, j in ij]
        # S_ij = [1 if y[i] > y[j] else -1 if y[i] < y[j] else 0 for i, j in ij]
        # s_ij, S_ij = np.array(s_ij), np.array(S_ij)
        #
        # lambda_ij = 0.5 * sigma * (1 - S_ij) - 1 / (1 + np.exp(sigma * s_ij))
        # for k, (i, j) in enumerate(ij):
        #     grad_input[i] += lambda_ij[k]
        #     grad_input[j] -= lambda_ij[k]

        if self.size_average:
            grad_input /= 0.5 * n_samples * (n_samples + 1)

        return grad_input
