import pytest
import numpy as np
import torch
from gradience.tensor import Tensor


def test_reshape_forward_backward():
    x_np = np.random.randn(2, 3, 4).astype(np.float64)
    x_gr = Tensor(x_np, requires_grad=True)
    x_pt = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)

    y_gr = x_gr.reshape(3, 8)
    y_pt = x_pt.reshape(3, 8)

    assert np.allclose(y_gr.data, y_pt.detach().numpy())

    loss_gr = y_gr.sum()
    loss_pt = y_pt.sum()
    loss_gr.backward()
    loss_pt.backward()

    assert np.allclose(x_gr.grad, x_pt.grad.numpy())


def test_reshape_with_negative_one():
    x_np = np.random.randn(2, 3, 4).astype(np.float64)
    x_gr = Tensor(x_np, requires_grad=True)
    x_pt = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)

    y_gr = x_gr.reshape(-1)
    y_pt = x_pt.reshape(-1)

    assert np.allclose(y_gr.data, y_pt.detach().numpy())

    loss_gr = y_gr.sum()
    loss_pt = y_pt.sum()
    loss_gr.backward()
    loss_pt.backward()

    assert np.allclose(x_gr.grad, x_pt.grad.numpy())


def test_flatten_forward_backward():
    x_np = np.random.randn(2, 3, 4).astype(np.float64)
    x_gr = Tensor(x_np, requires_grad=True)
    x_pt = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)

    y_gr = x_gr.flatten(start_dim=1)
    y_pt = torch.flatten(x_pt, start_dim=1)

    assert np.allclose(y_gr.data, y_pt.detach().numpy())

    loss_gr = y_gr.sum()
    loss_pt = y_pt.sum()
    loss_gr.backward()
    loss_pt.backward()

    assert np.allclose(x_gr.grad, x_pt.grad.numpy())
