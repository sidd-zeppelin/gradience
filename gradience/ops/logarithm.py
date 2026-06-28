import numpy as np
from gradience.autograd.function import Function

class LogOp(Function):
    
    @staticmethod
    def forward(ctx, x, base=np.e):
        ctx.save_for_backward(x)
        ctx.base = base
        if base == np.e:
            return np.log(x)
        return np.log(x) / np.log(base)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        base = ctx.base
        if base == np.e:
            return (grad_output / x,)
        return (grad_output / (x * np.log(base)),)
