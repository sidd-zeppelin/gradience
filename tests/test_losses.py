import numpy as np

from gradience.tensor import Tensor
from gradience.nn.losses.mse import MSELoss
from gradience.nn.losses.l1 import L1Loss
from gradience.nn.losses.cross_entropy import CrossEntropyLoss
from gradience.nn.losses.bce_with_logits import BCEWithLogitsLoss


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


def test_l1_loss():
    loss_fn = L1Loss()
    
    pred = Tensor(np.array([-1.0, 2.0, 3.0]), requires_grad=True)
    target = Tensor(np.array([1.0, 3.0, 3.0]))
    
    loss = loss_fn(pred, target)
    loss.backward()
    
    # | -1 - 1 | = 2
    # | 2 - 3 | = 1
    # | 3 - 3 | = 0
    # mean = 1.0
    assert np.allclose(loss.data, 1.0)
    
    # gradient: sign(pred - target) / N
    # sign(-2) / 3 = -1/3
    # sign(-1) / 3 = -1/3
    # sign(0) / 3 = 0
    np.testing.assert_allclose(pred.grad, np.array([-1/3, -1/3, 0]))


def test_cross_entropy_loss_indices():
    loss_fn = CrossEntropyLoss()
    
    # 2 samples, 3 classes
    pred = Tensor(np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]]), requires_grad=True)
    target = Tensor(np.array([0, 1]))
    
    loss = loss_fn(pred, target)
    loss.backward()
    
    assert loss.data > 0
    assert pred.grad.shape == (2, 3)
    # Sum of gradients for each sample should be roughly 0
    assert np.allclose(pred.grad.sum(axis=1), 0, atol=1e-6)

def test_cross_entropy_loss_onehot():
    loss_fn = CrossEntropyLoss()
    
    pred = Tensor(np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]]), requires_grad=True)
    target = Tensor(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    
    loss = loss_fn(pred, target)
    loss.backward()
    
    assert loss.data > 0
    assert pred.grad.shape == (2, 3)

def test_bce_with_logits_loss():
    loss_fn = BCEWithLogitsLoss()
    
    pred = Tensor(np.array([0.5, -0.5, 1.5]), requires_grad=True)
    target = Tensor(np.array([1.0, 0.0, 1.0]))
    
    loss = loss_fn(pred, target)
    loss.backward()
    
    assert loss.data > 0
    assert pred.grad.shape == (3,)
