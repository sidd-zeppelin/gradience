from gradience.autograd.function import Function

class NegationOp(Function):
    @staticmethod
    def forward(ctx, x):
        return (-x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output