class SGD(object):
    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters
        self.alpha = alpha

    def zero(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad *= 0

    def step(self, zero=True):
        for p in self.parameters:
            if p.grad is not None:
                p.data -= p.grad * self.alpha
                if(zero):
                    p.grad *= 0
