import pytest
import numpy as np
from gradience.tensor import Tensor
from gradience.nn.parameter import Parameter

def test_parameter_creation():
    p = Parameter([1.0, 2.0, 3.0])
    
    assert isinstance(p, Parameter)
    assert isinstance(p, Tensor)
    assert p.requires_grad is True
    assert p.shape == (3,)
    assert p.dtype.name == "float64"

def test_parameter_explicit_requires_grad():
    p = Parameter([1.0, 2.0], requires_grad=False)
    assert p.requires_grad is False

def test_parameter_repr():
    p = Parameter(5.0)
    rep = repr(p)
    assert rep.startswith("Parameter containing:\nTensor(")

def test_parameter_operations():
    p = Parameter([1.0, 2.0])
    t = Tensor([3.0, 4.0])
    
    result = p + t
    assert isinstance(result, Tensor)
    assert not isinstance(result, Parameter)
    assert result.requires_grad is True
    
    np.testing.assert_array_equal(result.data, np.array([4.0, 6.0]))

def test_parameter_backward():
    p = Parameter([2.0])
    t = Tensor([3.0])
    
    out = p * t
    out.backward()
    
    assert p.grad is not None
    assert t.grad is None  

def test_parameter_with_dtype():
    p = Parameter([1, 2], dtype=np.float32)
    assert p.dtype == np.float32

def test_parameter_clone():
    p = Parameter([1.0])
    p2 = p.clone()
    
    assert isinstance(p2, Tensor)
    assert p2.requires_grad is True
    np.testing.assert_array_equal(p.data, p2.data)
