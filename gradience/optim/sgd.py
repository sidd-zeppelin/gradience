import numpy as np
from gradience.optim.optimizer import Optimizer

class SGD(Optimizer):

    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocities = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            d_p = param.grad.copy()
            if self.weight_decay != 0:
                d_p += self.weight_decay * param.data
                
            if self.momentum != 0:
                buf = self.velocities[i]
                buf = self.momentum * buf + d_p
                self.velocities[i] = buf
                
                if self.nesterov:
                    d_p = d_p + self.momentum * buf
                else:
                    d_p = buf
                    
            param._data -= self.lr * d_p