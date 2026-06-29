# pyrefly: ignore [missing-import]
from gradience.autograd.function import Function
from gradience.utils.broadcast import unbroadcast

class DivisionOp(Function):
    
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return (x / y)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        
        grad_x = unbroadcast(grad_output / y, x.shape)
        grad_y = unbroadcast(grad_output * -x / (y * y), y.shape)
        
        return grad_x, grad_y