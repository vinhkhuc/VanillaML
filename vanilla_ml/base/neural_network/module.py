from abc import ABCMeta, abstractmethod


class Module(object):
    """
    Abstract Module class for neural net
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.output     = None
        self.grad_input = None

    @abstractmethod
    def fprop(self, input_data):
        self.output = input_data
        return self.output

    @abstractmethod
    def bprop(self, input_data, grad_output):
        self.grad_input = grad_output
        return self.grad_input

    @abstractmethod
    def update(self, params):
        pass

    @abstractmethod
    def share(self, m):
        pass

