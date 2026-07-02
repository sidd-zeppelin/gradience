import pytest
import numpy as np

from gradience.tensor import Tensor
from gradience.optim.optimizer import Optimizer
from gradience.optim.sgd import SGD
from gradience.nn.parameter import Parameter
from gradience.nn.layers.linear import Linear


def test_optimizer_base_class():
    param = Parameter(np.array([1.0]))
    opt = Optimizer([param])
    
    with pytest.raises(NotImplementedError):
        opt.step()


def test_sgd_step():
    param = Parameter(np.array([1.0, 2.0]))
    param.grad = np.array([0.5, -0.5])
    
    opt = SGD([param], lr=0.1)
    opt.step()
    
    # 1.0 - 0.1 * 0.5 = 0.95
    # 2.0 - 0.1 * -0.5 = 2.05
    assert np.allclose(param.data, np.array([0.95, 2.05]))


def test_sgd_zero_grad():
    param = Parameter(np.array([1.0, 2.0]))
    param.grad = np.array([0.5, -0.5])
    
    opt = SGD([param], lr=0.1)
    opt.zero_grad()
    
    assert param.grad is None


def test_sgd_with_module():
    layer = Linear(2, 2)
    opt = SGD(layer.parameters(), lr=0.1)
    
    x = Tensor(np.array([[1.0, 2.0]]))
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    # Check that step executes without crashing
    opt.step()
    opt.zero_grad()


def test_sgd_no_grad():
    param1 = Parameter(np.array([1.0, 2.0]))
    param2 = Parameter(np.array([1.0, 2.0]))
    
    param1.grad = np.array([0.5, -0.5])
    # param2 has no grad
    
    opt = SGD([param1, param2], lr=0.1)
    opt.step()
    
    # param1 should be updated
    assert np.allclose(param1.data, np.array([0.95, 2.05]))
    
    # param2 should be unchanged
    assert np.allclose(param2.data, np.array([1.0, 2.0]))
