import numpy as np
from gradience.autograd.function import Function

class SumOp(Function):
    
    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        ctx.save_for_backward(x)
        ctx.axis = axis
        ctx.keepdims = keepdims
        return np.sum(x, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        axis = ctx.axis
        keepdims = ctx.keepdims
        
        if axis is not None and not keepdims:
            grad_output = np.expand_dims(grad_output, axis=axis)
            
        return (np.broadcast_to(grad_output, x.shape),)
