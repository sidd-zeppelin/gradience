import numpy as np
from gradience.autograd.function import Function
from gradience.utils.convolution import compute_output_shape


class MaxPool2DOp(Function):

    @staticmethod
    def forward(ctx, x, kernel_size, stride=None, padding=0):
        # x is a NumPy array of shape (N, C, H, W)
        if stride is None:
            stride = kernel_size

        N, C, H, W = x.shape
        kh = kw = kernel_size

        H_out, W_out = compute_output_shape(H, W, kh, kw, stride, padding)

        # Pad input with -inf
        if padding > 0:
            pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))
            x_padded = np.pad(x, pad_width, mode="constant", constant_values=-np.inf)
        else:
            x_padded = x

        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        max_indices = np.zeros((N, C, H_out, W_out, 2), dtype=np.int32)

        for n in range(N):
            for c in range(C):
                for r in range(H_out):
                    for c_out in range(W_out):
                        h_start = r * stride
                        w_start = c_out * stride
                        patch = x_padded[n, c, h_start : h_start + kh, w_start : w_start + kw]
                        idx = np.argmax(patch)
                        patch_r = idx // kw
                        patch_c = idx % kw

                        out[n, c, r, c_out] = patch[patch_r, patch_c]
                        max_indices[n, c, r, c_out] = [h_start + patch_r, w_start + patch_c]

        ctx.x_shape = x.shape
        ctx.max_indices = max_indices
        ctx.padding = padding
        ctx.stride = stride
        ctx.kernel_size = kernel_size

        return out

    @staticmethod
    def backward(ctx, grad_output):
        N, C, H, W = ctx.x_shape
        max_indices = ctx.max_indices
        padding = ctx.padding

        H_out, W_out = grad_output.shape[2], grad_output.shape[3]

        h_padded = H + 2 * padding
        w_padded = W + 2 * padding
        grad_x_padded = np.zeros((N, C, h_padded, w_padded), dtype=grad_output.dtype)

        for n in range(N):
            for c in range(C):
                for r in range(H_out):
                    for c_out in range(W_out):
                        go = grad_output[n, c, r, c_out]
                        h_idx, w_idx = max_indices[n, c, r, c_out]
                        grad_x_padded[n, c, h_idx, w_idx] += go

        if padding > 0:
            grad_x = grad_x_padded[:, :, padding : -padding, padding : -padding]
        else:
            grad_x = grad_x_padded

        return (grad_x,)
