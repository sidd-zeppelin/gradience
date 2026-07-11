import numpy as np
from gradience.autograd.function import Function


class ReshapeOp(Function):

    @staticmethod
    def forward(ctx, x, shape):
        ctx.save_for_backward(x.shape)
        return x.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        orig_shape, = ctx.saved_tensors
        return (grad_output.reshape(orig_shape),)
