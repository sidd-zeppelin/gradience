import numpy as np
from gradience.autograd.function import Function

class SqrtOp(Function):
    
    @staticmethod
    def forward(ctx, x):
        result = np.sqrt(x)
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return (grad_output / (2.0 * result),)
