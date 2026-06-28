import numpy as np
from gradience.autograd.function import Function

class ATanOp(Function):
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.arctan(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return (grad_output / (1.0 + x**2),)
