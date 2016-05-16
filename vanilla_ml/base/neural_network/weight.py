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
        self.grad = np.zeros(sz, np.float32)

    def update(self, params):
        """
        Update weights
        """
        max_grad_norm = params.get('max_grad_norm')
        if max_grad_norm and max_grad_norm > 0:
            grad_norm = np.linalg.norm(self.grad, 2)
            if grad_norm > max_grad_norm:
                self.grad = self.grad * max_grad_norm / grad_norm

        self.D -= params['lrate'] * self.grad

        print("grad_w = %s" % self.grad)

        self.grad[:] = 0

    def clone(self):
        m = Weight(self.sz)
        m.D = np.copy(self.D)
        m.grad = np.copy(self.grad)
        return m
