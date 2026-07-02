import numpy as np
from gradience.optim.optimizer import Optimizer

class Adagrad(Optimizer):
    def __init__(self, parameters, lr=0.01, eps=1e-10, weight_decay=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.sum_squares = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
                
            grad = param.grad.copy()
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
                
            self.sum_squares[i] += grad ** 2
            
            param._data -= self.lr * grad / (np.sqrt(self.sum_squares[i]) + self.eps)
