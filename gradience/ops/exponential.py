import numpy as np
from gradience.autograd.function import Function

class ExpOp(Function):
    
    @staticmethod
    def forward(ctx, x):
        result = np.exp(x)
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return (grad_output * result,)
