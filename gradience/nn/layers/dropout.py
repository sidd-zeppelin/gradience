import numpy as np
from gradience.nn.module import Module
from gradience.tensor import Tensor

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
            
        mask = np.random.binomial(1, 1 - self.p, size=x.shape)
        mask = mask / (1 - self.p)
        return x * mask
