import numpy as np
from gradience.autograd.function import Function

class TanhOp(Function):
    
    @staticmethod
    def forward(ctx, x):
        t = np.tanh(x)
        ctx.save_for_backward(t)
        return t
    
    @staticmethod
    def backward(ctx, grad_output):
        t, = ctx.saved_tensors
        return (grad_output * (1.0 - t**2),)
