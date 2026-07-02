import numpy as np

from gradience.tensor import Tensor
from gradience.nn.activations.relu import ReLU
from gradience.nn.activations.sigmoid import Sigmoid
from gradience.nn.activations.tanh import Tanh


def test_relu_activation():
    layer = ReLU()
    x = Tensor(np.array([-1.0, 0.0, 1.0]))
    y = layer(x)
    
    assert np.all(y.data == np.array([0.0, 0.0, 1.0]))


def test_sigmoid_activation():
    layer = Sigmoid()
    x = Tensor(np.array([0.0]))
    y = layer(x)
    
    assert np.allclose(y.data, np.array([0.5]))
    

def test_tanh_activation():
    layer = Tanh()
    x = Tensor(np.array([0.0]))
    y = layer(x)
    
    assert np.allclose(y.data, np.array([0.0]))


def test_activation_repr():
    layer = ReLU()
    assert repr(layer) == "ReLU()"


def test_activation_backward():
    layer = ReLU()
    x = Tensor(np.array([-1.0, 1.0]), requires_grad=True)
    y = layer(x)
    
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
    assert np.all(x.grad == np.array([0.0, 1.0]))
