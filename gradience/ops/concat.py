import numpy as np
from gradience.autograd.function import Function


class ConcatOp(Function):

    @staticmethod
    def forward(ctx, *args, axis=1):
        ctx.axis = axis
        ctx.shapes = [x.shape for x in args]
        return np.concatenate(args, axis=axis)

    @staticmethod
    def backward(ctx, grad_output):
        axis = ctx.axis
        shapes = ctx.shapes

        grads = []
        start = 0
        for shape in shapes:
            size = shape[axis]
            indices = [slice(None)] * grad_output.ndim
            indices[axis] = slice(start, start + size)
            grads.append(grad_output[tuple(indices)])
            start += size

        return tuple(grads)
