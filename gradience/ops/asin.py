import numpy as np
from gradience.autograd.function import Function

class ASinOp(Function):
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.arcsin(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return (grad_output / np.sqrt(1.0 - x**2),)
