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

    def __init__(self):
        self.size_average = True

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
