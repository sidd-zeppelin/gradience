import numpy as np
from gradience.autograd.function import Function


class AdaptiveAvgPool2DOp(Function):

    @staticmethod
    def forward(ctx, x, output_size):
        if isinstance(output_size, int):
            H_out = W_out = output_size
        else:
            H_out, W_out = output_size

        N, C, H_in, W_in = x.shape
        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)

        ctx.x_shape = x.shape
        ctx.output_size = (H_out, W_out)

        for i in range(H_out):
            start_h = int(np.floor(i * H_in / H_out))
            end_h = int(np.ceil((i + 1) * H_in / H_out))
            for j in range(W_out):
                start_w = int(np.floor(j * W_in / W_out))
                end_w = int(np.ceil((j + 1) * W_in / W_out))

                patch = x[:, :, start_h:end_h, start_w:end_w]
                out[:, :, i, j] = np.mean(patch, axis=(2, 3))

        return out

    @staticmethod
    def backward(ctx, grad_output):
        N, C, H_in, W_in = ctx.x_shape
        H_out, W_out = ctx.output_size

        grad_x = np.zeros((N, C, H_in, W_in), dtype=grad_output.dtype)

        for i in range(H_out):
            start_h = int(np.floor(i * H_in / H_out))
            end_h = int(np.ceil((i + 1) * H_in / H_out))
            for j in range(W_out):
                start_w = int(np.floor(j * W_in / W_out))
                end_w = int(np.ceil((j + 1) * W_in / W_out))

                kh = end_h - start_h
                kw = end_w - start_w
                N_pixels = kh * kw

                go = grad_output[:, :, i, j]
                grad_x[:, :, start_h:end_h, start_w:end_w] += go[:, :, np.newaxis, np.newaxis] / N_pixels

        return (grad_x,)
