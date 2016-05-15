import numpy as np

from module import Module
from weight import Weight


class FeedForward(Module):
    """
    Feed-forward layer
    """
    def __init__(self, input_size, output_size):
        super(FeedForward, self).__init__()
        self.weight = Weight((input_size, output_size))

    def fprop(self, input_data):
        self.output = np.dot(input_data, self.weight.D)
        return self.output

    def bprop(self, input_data, grad_output):
        # FIXME: input_data's shape = 100 x 3, grad_output's shape = 100 x 3
        self.grad_input = np.dot(grad_output, self.weight.grad.T)
        return self.grad_input

    def share(self, m):
        pass

    def update(self, params):
        pass
