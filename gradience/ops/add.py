from gradience.autograd.function import Function

class AddOp(Function):
    
    @staticmethod
    def forward(ctx, x, y):
        return x + y
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output