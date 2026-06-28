import numpy as np
from gradience.autograd.function import Function

class SigmoidOp(Function):
    
    @staticmethod
    def forward(ctx, x):
        s = 1.0 / (1.0 + np.exp(-x))
        ctx.save_for_backward(s)
        return s
    
    @staticmethod
    def backward(ctx, grad_output):
        s, = ctx.saved_tensors
        return (grad_output * s * (1.0 - s),)
