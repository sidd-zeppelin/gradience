import numpy as np

from gradience.nn.module import Module
from gradience.nn.parameter import Parameter
from gradience.nn import init

class Linear(Module):
    
    def __init__(
        self, 
        in_features,
        out_features,
        bias=True,
        *,
        weight_initializer=init.he_uniform,
        bias_initializer=init.zeros,
    ):
        super().__init__()

        if in_features <= 0:
            raise ValueError(
                "in_features must be a positive integer."
            )

        if out_features <= 0:
            raise ValueError(
                "out_features must be a positive integer."
            )
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(
            np.empty(
                (in_features, out_features)
            )
        )
        
        weight_initializer(self.weight)
        
        if bias:
            self.bias = Parameter(
                np.empty(out_features)
            )
            bias_initializer(self.bias) 
        else:
            self.bias = None
    
    def forward(self, x):
        output = x @ self.weight
        
        if self.bias is not None:
            output = output + self.bias
        
        return output

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def __repr__(self):
        return (
            f"Linear("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
            f")"
        )