import numpy as np

from module import Module


class Sigmoid(Module):

    def fprop(self, input_data):
        self.output = 1. / (1 + np.exp(-input_data))
        return self.output

    def bprop(self, input_data, grad_output):
        return grad_output * self.output * (1. - self.output)

    def update(self, params):
        pass

    def share(self, m):
        pass


class Softmax(Module):

    def __init__(self, skip_bprop=False):
        super(Softmax, self).__init__()
        self.skip_bprop = skip_bprop  # for the output module

    # FIXME: Is input_data's values supposed to changed inside fprob?
    def fprop(self, input_data):
        input_data -= np.max(input_data, axis=0)
        input_data += 1.0

        a = np.exp(input_data)
        sum_a = a.sum(axis=0)

        self.output = a / sum_a[None, :]  # divide by row
        return self.output

    def bprop(self, input_data, grad_output):
        if not self.skip_bprop:
            z = grad_output - np.sum(self.output * grad_output, axis=0)
            self.grad_input = self.output * z
        else:
            self.grad_input = grad_output

        return self.grad_input

    def update(self, params):
        pass

    def share(self, m):
        pass
