import numpy as np
from gradience.optim.optimizer import Optimizer

class RMSprop(Optimizer):
    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.b = [np.zeros_like(p.data) for p in self.parameters] # momentum buffer

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
                
            grad = param.grad.copy()
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data
                
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (grad ** 2)
            avg = self.v[i]
            
            if self.momentum > 0:
                self.b[i] = self.momentum * self.b[i] + grad / (np.sqrt(avg) + self.eps)
                param._data -= self.lr * self.b[i]
            else:
                param._data -= self.lr * grad / (np.sqrt(avg) + self.eps)
