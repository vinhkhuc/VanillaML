import numpy as np

from module import Module


class ReLU(Module):
    """ ReLU module """

    def fprop(self, input_data):
        self.output = np.multiply(input_data, input_data > 0)
        return self.output

    def bprop(self, input_data, grad_output):
        self.grad_input = np.multiply(grad_output, input_data > 0)
        return self.grad_input

    def update(self, params):
        pass

    def share(self, m):
        pass


class Sigmoid(Module):

    def fprop(self, input_data):
        self.output = 1. / (1 + np.exp(-input_data))
        return self.output

    def bprop(self, input_data, grad_output):
        return np.multiply(grad_output, np.multiply(self.output, 1. - self.output))
        # return grad_output * self.output * (1. - self.output)

    def update(self, params):
        pass

    def share(self, m):
        pass


class Softmax(Module):

    def __init__(self, skip_bprop=False):
        super(Softmax, self).__init__()
        self.skip_bprop = skip_bprop  # for the output module

    def fprop(self, input_data):
        temp_data = np.copy(input_data)
        temp_data -= np.max(temp_data, axis=1)[:, None]
        # temp_data += 1.0   # FIXME: Is it correct?

        a = np.exp(temp_data)
        sum_a = a.sum(axis=1)

        self.output = a / sum_a[:, None]  # divide by row
        return self.output

    def bprop(self, input_data, grad_output):
        if not self.skip_bprop:
            z = grad_output - np.sum(self.output * grad_output, axis=1)  # FIXME: Is axis=1 correct?
            self.grad_input = self.output * z
        else:
            self.grad_input = grad_output

        return self.grad_input

    def update(self, params):
        pass

    def share(self, m):
        pass
