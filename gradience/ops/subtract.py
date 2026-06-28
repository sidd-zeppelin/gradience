from gradience.autograd.function import Function

class SubtractOp(Function):
    
    @staticmethod
    def forward(ctx, x, y):
        return x - y
        
    def backward(ctx, grad_output):
        return grad_output, grad_output