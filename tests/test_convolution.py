import pytest
import numpy as np
import torch
from gradience.tensor import Tensor
from gradience.nn.convolution.conv2d import Conv2D
from gradience.utils.convolution import compute_output_shape, pad_input, extract_patch


def test_compute_output_shape():
    assert compute_output_shape(8, 8, 3, 3, 1, 0) == (6, 6)
    assert compute_output_shape(8, 8, 3, 3, 1, 1) == (8, 8)
    assert compute_output_shape(8, 8, 3, 3, 2, 1) == (4, 4)
    assert compute_output_shape(8, 8, 1, 1, 1, 0) == (8, 8)

    with pytest.raises(ValueError):
        compute_output_shape(-8, 8, 3, 3, 1, 1)
    with pytest.raises(ValueError):
        compute_output_shape(8, 8, 3, 3, -1, 1)
    with pytest.raises(ValueError):
        compute_output_shape(8, 8, 3, 3, 1, -1)
    with pytest.raises(ValueError):
        compute_output_shape(2, 2, 3, 3, 1, 0)


def test_pad_input():
    x = np.ones((2, 3, 4, 4))
    padded = pad_input(x, 1)
    assert padded.shape == (2, 3, 6, 6)
    assert np.all(padded[:, :, 0, :] == 0)
    assert np.all(padded[:, :, -1, :] == 0)
    assert np.all(padded[:, :, :, 0] == 0)
    assert np.all(padded[:, :, :, -1] == 0)
    assert np.all(padded[:, :, 1:5, 1:5] == 1)


def test_extract_patch():
    x = np.arange(1, 17).reshape((1, 1, 4, 4))
    patch = extract_patch(x, 0, 1, 1, 2, 2)
    expected = np.array([[[6, 7], [10, 11]]])
    assert np.array_equal(patch, expected)


def test_conv2d_match_pytorch_bias():
    n, c, h, w = 2, 3, 6, 6
    out_channels = 4
    kernel_size = 3
    stride = 2
    padding = 1

    x_np = np.random.randn(n, c, h, w).astype(np.float64)
    w_np = np.random.randn(out_channels, c, kernel_size, kernel_size).astype(np.float64)
    b_np = np.random.randn(out_channels).astype(np.float64)

    x_pt = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)
    conv_pt = torch.nn.Conv2d(c, out_channels, kernel_size, stride=stride, padding=padding, bias=True)
    conv_pt.weight.data = torch.tensor(w_np, dtype=torch.float64)
    conv_pt.bias.data = torch.tensor(b_np, dtype=torch.float64)

    y_pt = conv_pt(x_pt)
    loss_pt = y_pt.sum()
    loss_pt.backward()

    x_gr = Tensor(x_np, requires_grad=True)
    conv_gr = Conv2D(c, out_channels, kernel_size, stride=stride, padding=padding, bias=True)
    conv_gr.weight.data[...] = w_np
    conv_gr.bias.data[...] = b_np

    y_gr = conv_gr(x_gr)
    loss_gr = y_gr.sum()
    loss_gr.backward()

    assert np.allclose(y_gr.data, y_pt.detach().numpy(), atol=1e-9)
    assert np.allclose(x_gr.grad, x_pt.grad.numpy(), atol=1e-9)
    assert np.allclose(conv_gr.weight.grad, conv_pt.weight.grad.numpy(), atol=1e-9)
    assert np.allclose(conv_gr.bias.grad, conv_pt.bias.grad.numpy(), atol=1e-9)


def test_conv2d_match_pytorch_no_bias():
    n, c, h, w = 2, 2, 5, 5
    out_channels = 3
    kernel_size = 3
    stride = 1
    padding = 0

    x_np = np.random.randn(n, c, h, w).astype(np.float64)
    w_np = np.random.randn(out_channels, c, kernel_size, kernel_size).astype(np.float64)

    x_pt = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)
    conv_pt = torch.nn.Conv2d(c, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
    conv_pt.weight.data = torch.tensor(w_np, dtype=torch.float64)

    y_pt = conv_pt(x_pt)
    loss_pt = y_pt.sum()
    loss_pt.backward()

    x_gr = Tensor(x_np, requires_grad=True)
    conv_gr = Conv2D(c, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
    conv_gr.weight.data[...] = w_np

    y_gr = conv_gr(x_gr)
    loss_gr = y_gr.sum()
    loss_gr.backward()

    assert np.allclose(y_gr.data, y_pt.detach().numpy(), atol=1e-9)
    assert np.allclose(x_gr.grad, x_pt.grad.numpy(), atol=1e-9)
    assert np.allclose(conv_gr.weight.grad, conv_pt.weight.grad.numpy(), atol=1e-9)


def test_conv2d_various_kernel_sizes():
    configurations = [
        (1, 1, 0, True),
        (3, 1, 1, True),
        (3, 2, 1, False),
        (1, 2, 0, False)
    ]

    n, c, h, w = 1, 2, 6, 6
    out_channels = 2

    for kernel_size, stride, padding, bias in configurations:
        x_np = np.random.randn(n, c, h, w).astype(np.float64)
        w_np = np.random.randn(out_channels, c, kernel_size, kernel_size).astype(np.float64)

        x_pt = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)
        conv_pt = torch.nn.Conv2d(c, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        conv_pt.weight.data = torch.tensor(w_np, dtype=torch.float64)
        if bias:
            b_np = np.random.randn(out_channels).astype(np.float64)
            conv_pt.bias.data = torch.tensor(b_np, dtype=torch.float64)

        y_pt = conv_pt(x_pt)
        loss_pt = y_pt.sum()
        loss_pt.backward()

        x_gr = Tensor(x_np, requires_grad=True)
        conv_gr = Conv2D(c, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        conv_gr.weight.data[...] = w_np
        if bias:
            conv_gr.bias.data[...] = b_np

        y_gr = conv_gr(x_gr)
        loss_gr = y_gr.sum()
        loss_gr.backward()

        assert np.allclose(y_gr.data, y_pt.detach().numpy(), atol=1e-9)
        assert np.allclose(x_gr.grad, x_pt.grad.numpy(), atol=1e-9)
        assert np.allclose(conv_gr.weight.grad, conv_pt.weight.grad.numpy(), atol=1e-9)
        if bias:
            assert np.allclose(conv_gr.bias.grad, conv_pt.bias.grad.numpy(), atol=1e-9)
