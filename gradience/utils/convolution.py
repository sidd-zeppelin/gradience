import numpy as np


def compute_output_shape(h_in, w_in, kh, kw, stride, padding):
    if h_in <= 0 or w_in <= 0 or kh <= 0 or kw <= 0 or stride <= 0:
        raise ValueError("Dimensions and stride must be positive integers.")
    if padding < 0:
        raise ValueError("Padding must be a non-negative integer.")
    if h_in - kh + 2 * padding < 0:
        raise ValueError("Kernel size is too large for the input height with the given padding.")
    if w_in - kw + 2 * padding < 0:
        raise ValueError("Kernel size is too large for the input width with the given padding.")
    
    h_out = (h_in - kh + 2 * padding) // stride + 1
    w_out = (w_in - kw + 2 * padding) // stride + 1
    return h_out, w_out


def pad_input(x, padding):
    pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))
    return np.pad(x, pad_width, mode="constant", constant_values=0)


def extract_patch(x_padded, batch_idx, start_row, start_col, kh, kw):
    return x_padded[batch_idx, :, start_row : start_row + kh, start_col : start_col + kw]
