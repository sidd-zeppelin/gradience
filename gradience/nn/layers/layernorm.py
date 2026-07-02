import numpy as np
from gradience.nn.module import Module
from gradience.nn.parameter import Parameter
from gradience.tensor import Tensor

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        
        self.weight = Parameter(np.ones(self.normalized_shape))
        self.bias = Parameter(np.zeros(self.normalized_shape))
        
    def forward(self, x):
        dims = tuple(range(len(x.shape) - len(self.normalized_shape), len(x.shape)))
        
        mean = x.mean(axis=dims, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=dims, keepdims=True)
        
        x_hat = (x - mean) / ((var + self.eps) ** 0.5)
        return x_hat * self.weight + self.bias
