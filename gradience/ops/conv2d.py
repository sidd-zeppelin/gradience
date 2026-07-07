import numpy as np
from gradience.autograd.function import Function
from gradience.utils.convolution import compute_output_shape, pad_input, extract_patch


class Conv2DOp(Function):

    @staticmethod
    def forward(ctx, x, weight, *args, stride=1, padding=0):
        bias = args[0] if len(args) > 0 else None

        ctx.stride = stride
        ctx.padding = padding
        ctx.has_bias = bias is not None

        x_padded = pad_input(x, padding)
        ctx.save_for_backward(x_padded, weight)

        N, C, H, W = x.shape
        F, _, kh, kw = weight.shape
        H_out, W_out = compute_output_shape(H, W, kh, kw, stride, padding)

        out = np.zeros((N, F, H_out, W_out), dtype=x.dtype)

        for n in range(N):
            for f in range(F):
                for r in range(H_out):
                    for c in range(W_out):
                        patch = extract_patch(x_padded, n, r * stride, c * stride, kh, kw)
                        val = np.sum(patch * weight[f])
                        if bias is not None:
                            val += bias[f]
                        out[n, f, r, c] = val

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_padded, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        has_bias = ctx.has_bias

        N, C, H_padded, W_padded = x_padded.shape
        F, _, kh, kw = weight.shape
        _, _, H_out, W_out = grad_output.shape

        grad_x_padded = np.zeros_like(x_padded)
        grad_weight = np.zeros_like(weight)

        if has_bias:
            grad_bias = np.zeros(F, dtype=grad_output.dtype)

        for n in range(N):
            for f in range(F):
                for r in range(H_out):
                    for c in range(W_out):
                        go = grad_output[n, f, r, c]
                        h_start = r * stride
                        w_start = c * stride

                        grad_weight[f] += go * x_padded[n, :, h_start : h_start + kh, w_start : w_start + kw]
                        grad_x_padded[n, :, h_start : h_start + kh, w_start : w_start + kw] += go * weight[f]

                        if has_bias:
                            grad_bias[f] += go

        if padding > 0:
            grad_x = grad_x_padded[:, :, padding : -padding, padding : -padding]
        else:
            grad_x = grad_x_padded

        if has_bias:
            return grad_x, grad_weight, grad_bias
        else:
            return grad_x, grad_weight
