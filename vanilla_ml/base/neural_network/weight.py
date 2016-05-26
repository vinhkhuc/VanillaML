import numpy as np


class Weight(object):
    def __init__(self, sz):
        """
        Initialize weight
        Args:
            sz (tuple): shape
        """
        self.sz = sz
        self.D = 0.1 * np.random.standard_normal(sz)
        # self.D = np.ones(sz)   # TODO: Remove this
        self.grad = np.zeros(sz, np.float32)

        # print("D = %s, grad = %s" % (self.D, self.grad))

    def update(self, params):
        """
        Update weights
        """
        max_grad_norm = params.get('max_grad_norm')
        if max_grad_norm and max_grad_norm > 0:
            grad_norm = np.linalg.norm(self.grad, 2)
            if grad_norm > max_grad_norm:
                print("Applying grad_norm ...")
                self.grad = self.grad * max_grad_norm / grad_norm

        # Regularization
        # self.grad += 10 * 2 * self.D

        self.D -= params['lrate'] * self.grad
        # print("D = %s, grad = %s" % (self.D, self.grad))

        self.grad[:] = 0

    def clone(self):
        m = Weight(self.sz)
        m.D = np.copy(self.D)
        m.grad = np.copy(self.grad)
        return m
