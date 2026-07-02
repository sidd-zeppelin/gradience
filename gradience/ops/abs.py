import numpy as np
from gradience.autograd.function import Function

class Abs(Function):
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.abs(x)
        
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return (grad_output * np.sign(x),)
