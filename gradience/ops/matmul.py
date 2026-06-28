import numpy as np
from gradience.autograd.function import Function

class MatMulOp(Function):
    
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return np.matmul(a, b)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        
        a_is_1d = a.ndim == 1
        b_is_1d = b.ndim == 1
        
        if a_is_1d and b_is_1d:
            # a: (K,), b: (K,) -> grad_output: ()
            grad_a = grad_output * b
            grad_b = grad_output * a
        elif a_is_1d:
            # a: (K,), b: (K, N) -> grad_output: (N,)
            grad_a = np.matmul(grad_output, np.swapaxes(b, -1, -2))
            grad_b = np.outer(a, grad_output)
        elif b_is_1d:
            # a: (M, K), b: (K,) -> grad_output: (M,)
            grad_a = np.outer(grad_output, b)
            grad_b = np.matmul(np.swapaxes(a, -1, -2), grad_output)
        else:
            # standard 2D or N-D
            grad_a = np.matmul(grad_output, np.swapaxes(b, -1, -2))
            grad_b = np.matmul(np.swapaxes(a, -1, -2), grad_output)
            
        return grad_a, grad_b
