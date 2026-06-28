import numpy as np
from gradience.autograd.function import Function

class MeanOp(Function):
    
    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        ctx.save_for_backward(x)
        ctx.axis = axis
        ctx.keepdims = keepdims
        return np.mean(x, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        axis = ctx.axis
        keepdims = ctx.keepdims
        
        if axis is not None and not keepdims:
            grad_output = np.expand_dims(grad_output, axis=axis)
            
        grad = np.broadcast_to(grad_output, x.shape)
        
        if axis is None:
            num_elements = x.size
        else:
            if isinstance(axis, tuple):
                num_elements = np.prod([x.shape[d] for d in axis])
            else:
                num_elements = x.shape[axis]
                
        return (grad / float(num_elements),)
