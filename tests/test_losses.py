import numpy as np

from gradience.tensor import Tensor
from gradience.nn.losses.mse import MSELoss


def test_mse_loss_forward():
    loss_fn = MSELoss()
    
    pred = Tensor(np.array([1.0, 2.0, 3.0]))
    target = Tensor(np.array([1.0, 2.0, 3.0]))
    
    loss = loss_fn(pred, target)
    assert np.allclose(loss.data, 0.0)
    
    pred2 = Tensor(np.array([1.0, 2.0, 3.0]))
    target2 = Tensor(np.array([2.0, 3.0, 4.0]))
    
    loss2 = loss_fn(pred2, target2)
    # (1^2 + 1^2 + 1^2) / 3 = 1.0
    assert np.allclose(loss2.data, 1.0)


def test_mse_loss_backward():
    loss_fn = MSELoss()
    
    pred = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    target = Tensor(np.array([2.0, 3.0, 4.0]))
    
    loss = loss_fn(pred, target)
    loss.backward()
    
    assert pred.grad is not None
    # d/d_pred ((pred - target)^2 / N) = 2 * (pred - target) / N
    # = 2 * (-1) / 3 = -2/3
    expected_grad = np.array([-2/3, -2/3, -2/3])
    assert np.allclose(pred.grad, expected_grad)


def test_mse_loss_repr():
    loss_fn = MSELoss()
    assert repr(loss_fn) == "MSELoss()"
