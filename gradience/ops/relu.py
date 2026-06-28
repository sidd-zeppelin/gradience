import numpy as np
from gradience.autograd.function import Function

class ReLUOp(Function):
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.maximum(0, x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return (grad_output * (x > 0),)
