import pytest
import numpy as np

from gradience.nn.layers.linear import Linear
from gradience.tensor import Tensor

def test_linear_initialization():
    
    layer = Linear(
        in_features=3,
        out_features=2,
    )
    
    assert layer.weight.shape == (3, 2)
    assert layer.bias.shape == (2, )
    
def test_linear_parameters():
    
    layer = Linear(3, 2)
    
    parameters = layer.parameters()
    
    assert len(parameters) == 2
    assert layer.weight in parameters
    assert layer.bias in parameters
    
def test_linear_forward():
    
    layer = Linear(3, 2)
    
    x = Tensor(
        np.random.randn(5, 3)
    )
    
    y = layer(x)
    
    assert y.shape == (5, 2)

def test_linear_no_bias():
    
    layer = Linear(3, 2, bias=False)
    
    assert layer.weight.shape == (3, 2)
    assert layer.bias is None
    
    parameters = layer.parameters()
    
    assert len(parameters) == 1
    assert layer.weight in parameters
    
    x = Tensor(np.random.randn(5, 3))
    y = layer(x)
    
    assert y.shape == (5, 2)

def test_linear_backward():
    
    layer = Linear(3, 2)
    
    x = Tensor(np.random.randn(5, 3))
    y = layer(x)
    
    loss = y.sum()
    loss.backward()
    
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == (3, 2)
    
    assert layer.bias.grad is not None
    assert layer.bias.grad.shape == (2, )

def test_linear_invalid_features():
    with pytest.raises(ValueError, match="in_features must be a positive integer."):
        Linear(0, 2)
        
    with pytest.raises(ValueError, match="out_features must be a positive integer."):
        Linear(3, -1)

def test_linear_custom_initializers():
    from gradience.nn import init
    
    layer = Linear(
        in_features=3,
        out_features=2,
        weight_initializer=init.ones,
        bias_initializer=init.ones
    )
    
    assert np.all(layer.weight.data == 1.0)
    assert np.all(layer.bias.data == 1.0)
