import itertools
from abc import ABCMeta, abstractmethod

import numpy as np

from vanilla_ml.util.metrics.ranking import delta_ndcg, delta_dcg


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
    """ Loss for RankNet (see the section 2 about RankNet in the paper
    "From RankNet to LambdaRank to LambdaMART: An Overview", Christ Burges).
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
        for i, j in itertools.combinations(range(n_samples), 2):
            s_ij = s[i] - s[j]
            S_ij = 1 if y[i] > y[j] else -1 if y[i] < y[j] else 0
            cost += 0.5 * sigma * (1 - S_ij) * s_ij + np.log(1 + np.exp(-sigma * s_ij))

        if self.size_average:
            cost /= 0.5 * n_samples * (n_samples + 1)  # normalized by the total number of pairs

        return cost

    def bprop(self, input_data, target_data):
        """ Back-propagation. Here we use the approach of calculating gradient
        as shown in the section 2.1 in the paper.

        Args:
            input_data:
            target_data:

        Returns:
            ndarray: gradient

        """
        s, y, sigma = input_data, target_data, self.sigma
        n_samples = s.shape[0]

        grad_input = np.zeros_like(input_data, np.float32)  # grad_input is the lambda in the paper)
        for i, j in itertools.combinations(range(n_samples), 2):
            S_ij = 1 if y[i] > y[j] else -1 if y[i] < y[j] else 0
            s_ij = s[i] - s[j]
            lambda_ij = 0.5 * sigma * (1 - S_ij) - sigma / (1 + np.exp(sigma * s_ij))  # dcost/ds_i
            grad_input[i] += lambda_ij
            grad_input[j] -= lambda_ij

        if self.size_average:
            grad_input /= 0.5 * n_samples * (n_samples + 1)

        return grad_input


class LambdaRankLoss(Loss):
    """ Loss for LambdaRank (see the equation 6 in the section 4.1 in the paper
    "From RankNet to LambdaRank to LambdaMART: An Overview", Christ Burges).
    """
    def __init__(self, sigma=1., size_average=True):
        self.sigma = sigma
        self.size_average = size_average

    def fprop(self, input_data, target_data):
        # Try to use the same notation as in the paper
        s, y, sigma = input_data.ravel(), target_data.ravel(), self.sigma

        # Iterate over all combinations of indices, i.e (0, 0), (0, 1), ...
        n_samples = s.shape[0]
        cost = 0
        for i, j in itertools.combinations(range(n_samples), 2):
            s_ij = s[i] - s[j]
            delta_ij = delta_dcg(y, s, i, j)
            # Similar to RankNetLoss but here we set S_ij = 1 and we multiply it by delta_ndcg
            # cost += np.log(1 + np.exp(-sigma * s_ij)) * abs(ndcg_delta_ij)

            S_ij = 1 if y[i] > y[j] else -1 if y[i] < y[j] else 0
            cost += (0.5 * sigma * (1 - S_ij) * s_ij + np.log(1 + np.exp(-sigma * s_ij))) * abs(delta_ij)

        if self.size_average:
            cost /= 0.5 * n_samples * (n_samples + 1)  # normalized by the total number of pairs

        return cost

    def bprop(self, input_data, target_data):
        """ Back-propagation for LambdaRank.

        Args:
            input_data (ndarray):
            target_data (ndarray):

        Returns:
            ndarray: gradient

        """
        s, y, sigma = input_data.ravel(), target_data.ravel(), self.sigma
        n_samples = s.shape[0]

        grad_input = np.zeros_like(input_data, np.float32)  # lambda
        for i, j in itertools.combinations(range(n_samples), 2):
            s_ij = s[i] - s[j]
            delta_ij = delta_dcg(y, s, i, j)
            # FIXME: ndcg_delta_ij is pretty small which almost doesn't any affect on gradient's updates.
            # TODO: Should we use delta_ndcg or delta_dcg?
            # In detla_ndcg, the ideal DCG must be computed for all ys (not just the training batch).

            # lambda_ij = -sigma / (1 + np.exp(sigma * s_ij)) * abs(ndcg_delta_ij)

            S_ij = 1 if y[i] > y[j] else -1 if y[i] < y[j] else 0
            lambda_ij = (0.5 * sigma * (1 - S_ij) - sigma / (1 + np.exp(sigma * s_ij))) * abs(delta_ij)

            # from vanilla_ml.util.metrics.ranking import idcg
            # idcg_score = idcg(y, k=len(y))
            # # idcg_score = 1.
            # dcg_delta_ij = (2 ** y[i] - 2 ** y[j]) * (1 / np.log2(2 + i) - 1 / np.log2(2 + j))
            # ndcg_delta_ij = dcg_delta_ij / idcg_score
            #
            # lambda_ij = sigma / (1 + np.exp(sigma * s_ij)) * ndcg_delta_ij

            # if i == 0 and j == 1:
            #     print("\tndcg_delta_ij = %g, lambda_ij = %g" % (ndcg_delta_ij, lambda_ij))

            grad_input[i] += lambda_ij
            grad_input[j] -= lambda_ij

        # print(grad_input[0])

        # In the paper, the optimizer maximizes LambdaRank's utility C.
        # But our optimizer always minimizes the target function.
        # grad_input = -grad_input

        if self.size_average:
            grad_input /= 0.5 * n_samples * (n_samples + 1)

        return grad_input
