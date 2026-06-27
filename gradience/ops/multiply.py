from gradience.autograd.function import Function
from gradience.autograd.context import Context

class MultOp(Function):
    
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return (x * y)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        
        grad_x = grad_output * y
        grad_y = grad_output * x
        
        return grad_x, grad_y