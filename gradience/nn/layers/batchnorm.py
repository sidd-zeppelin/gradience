import numpy as np
from gradience.nn.module import Module
from gradience.nn.parameter import Parameter
from gradience.tensor import Tensor

class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = Parameter(np.ones(num_features))
        self.bias = Parameter(np.zeros(num_features))
        
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
    def forward(self, x):
        if self.training:
            mean = x.mean(axis=0, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=0, keepdims=True)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data.flatten()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data.flatten()
            
            x_hat = (x - mean) / ((var + self.eps) ** 0.5)
        else:
            mean = self.running_mean.reshape(1, -1)
            var = self.running_var.reshape(1, -1)
            x_hat = (x - mean) / ((var + self.eps) ** 0.5)
            
        return x_hat * self.weight + self.bias
