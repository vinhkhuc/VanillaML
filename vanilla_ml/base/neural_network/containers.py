from module import Module


class Container(Module):
    """
    Container
    """
    def __init__(self):
        super(Container, self).__init__()
        self.modules = []

    def add(self, m):
        self.modules.append(m)

    def update(self, params):
        for module in self.modules:
            module.update(params)

    def share(self, m):
        for c_module, m_module in zip(self.modules, m.modules):
            c_module.share(m_module)

    def fprop(self, input_data):
        pass

    def bprop(self, input_data, grad_output):
        pass


class Sequential(Container):

    def fprop(self, input_data):
        temp = input_data
        for module in self.modules:
            temp = module.fprop(temp)

        self.output = temp
        return self.output

    def bprop(self, input_data, grad_output):
        for i in range(len(self.modules) - 1, 0, -1):
            grad_input = self.modules[i].bprop(self.modules[i - 1].output, grad_output)
            grad_output = grad_input
        grad_input = self.modules[0].bprop(input_data, grad_output)

        self.grad_input = grad_input
        return self.grad_input
