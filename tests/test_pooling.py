import pytest
import numpy as np
import torch
from gradience.tensor import Tensor
from gradience.nn.layers.pooling import MaxPool2D, AdaptiveAvgPool2D


def test_maxpool2d_forward_backward():
    n, c, h, w = 2, 3, 8, 8
    kernel_size = 3
    stride = 2
    padding = 1

    x_np = np.random.randn(n, c, h, w).astype(np.float64)

    # PyTorch setup
    x_pt = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)
    pool_pt = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    y_pt = pool_pt(x_pt)
    loss_pt = y_pt.sum()
    loss_pt.backward()

    # Gradience setup
    x_gr = Tensor(x_np, requires_grad=True)
    pool_gr = MaxPool2D(kernel_size=kernel_size, stride=stride, padding=padding)
    y_gr = pool_gr(x_gr)
    loss_gr = y_gr.sum()
    loss_gr.backward()

    # Verification
    assert np.allclose(y_gr.data, y_pt.detach().numpy())
    assert np.allclose(x_gr.grad, x_pt.grad.numpy())


def test_maxpool2d_no_padding():
    n, c, h, w = 1, 2, 6, 6
    kernel_size = 2
    stride = 2
    padding = 0

    x_np = np.random.randn(n, c, h, w).astype(np.float64)

    # PyTorch setup
    x_pt = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)
    pool_pt = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    y_pt = pool_pt(x_pt)
    loss_pt = y_pt.sum()
    loss_pt.backward()

    # Gradience setup
    x_gr = Tensor(x_np, requires_grad=True)
    pool_gr = MaxPool2D(kernel_size=kernel_size, stride=stride, padding=padding)
    y_gr = pool_gr(x_gr)
    loss_gr = y_gr.sum()
    loss_gr.backward()

    # Verification
    assert np.allclose(y_gr.data, y_pt.detach().numpy())
    assert np.allclose(x_gr.grad, x_pt.grad.numpy())


def test_adaptive_avgpool2d_forward_backward():
    n, c, h, w = 2, 3, 10, 10
    output_size = (4, 4)

    x_np = np.random.randn(n, c, h, w).astype(np.float64)

    # PyTorch setup
    x_pt = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)
    pool_pt = torch.nn.AdaptiveAvgPool2d(output_size)
    y_pt = pool_pt(x_pt)
    loss_pt = y_pt.sum()
    loss_pt.backward()

    # Gradience setup
    x_gr = Tensor(x_np, requires_grad=True)
    pool_gr = AdaptiveAvgPool2D(output_size)
    y_gr = pool_gr(x_gr)
    loss_gr = y_gr.sum()
    loss_gr.backward()

    # Verification
    assert np.allclose(y_gr.data, y_pt.detach().numpy())
    assert np.allclose(x_gr.grad, x_pt.grad.numpy())
