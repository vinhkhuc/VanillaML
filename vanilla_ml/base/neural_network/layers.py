import numpy as np

from module import Module
from weight import Weight


# class FeedForward(Module):
#     """
#     Feed-forward layer
#     """
#     def __init__(self, input_size, output_size):
#         super(FeedForward, self).__init__()
#         self.weight = Weight((input_size, output_size))
#
#     def fprop(self, input_data):
#         self.output = np.dot(input_data, self.weight.D)
#         return self.output
#
#     def bprop(self, input_data, grad_output):
#         self.weight.grad = input_data
#         self.grad_input = np.dot(grad_output, self.weight.grad.T)
#         return self.grad_input
#
#     def share(self, m):
#         pass
#
#     def update(self, params):
#         self.weight.update(params)

class Linear(Module):
    """
    Linear Layer (feed-forward layer)
    """
    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.weight  = Weight((in_dim, out_dim))
        self.bias    = Weight((1, out_dim))

    def fprop(self, input_data):
        # high_dimension_input = input_data.ndim > 2
        #
        # # Reshape input
        # if high_dimension_input:
        #     input_data = input_data.reshape(input_data.shape[0], -1)

        self.output = np.dot(input_data, self.weight.D) + self.bias.D

        # # Reshape output
        # if high_dimension_input:
        #     self.output = self.output.reshape(self.output.shape[0], -1)

        return self.output

    # TODO: Rename input_data -> prev_layer_output
    def bprop(self, input_data, grad_output):
        # orig_input_data_shape = input_data.shape
        # high_dimension_input = input_data.ndim > 2
        #
        # # Reshape input and grad_output
        # if high_dimension_input:
        #     input_data  = input_data.reshape(input_data.shape[0], -1)
        #     grad_output = grad_output.reshape(grad_output.shape[0], -1)

        # self.weight.grad = self.weight.grad + np.dot(input_data.T, grad_output)
        # self.bias.grad   = self.bias.grad + grad_output.sum(axis=0)
        self.weight.grad = np.dot(input_data.T, grad_output)
        self.bias.grad   = grad_output.sum(axis=0)
        self.grad_input  = np.dot(grad_output, self.weight.D.T)

        # if high_dimension_input:
        #     self.grad_input = self.grad_input.reshape(orig_input_data_shape)

        return self.grad_input

    def update(self, params):
        self.weight.update(params)
        self.bias.update(params)

    def share(self, m):
        self.weight = m.weight
        self.bias = m.bias
