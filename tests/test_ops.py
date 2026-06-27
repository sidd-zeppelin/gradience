import numpy as np

from gradience.tensor import Tensor

def test_addition():
    
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    
    z = x + y
    np.testing.assert_array_equal(
        z.data,
        np.array([5, 7, 9], dtype=np.float32)
    )
    
def test_multiplication():
    
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    
    z = x * y
    np.testing.assert_array_equal(
        z.data,
        np.array([4, 10, 18], dtype=np.float32)
    )
    
def test_scalar_addition():
    x = Tensor([1, 2, 3])
    y = x + 3
    
    np.testing.assert_array_equal(
        y.data,
        np.array([4, 5, 6], dtype=np.float32)
    )

def test_scalar_multiplication():
    x = Tensor([1, 2, 3])
    y = x * 5
    
    np.testing.assert_array_equal(
        y.data,
        np.array([5, 10, 15], dtype=np.float32)
    )
    
def test_requires_grad_propagation():
    x = Tensor([1, 2, 3], requires_grad= True)
    y = Tensor([4, 5, 6])
    
    z = x + y
    w = x * y
    
    assert z.requires_grad is True
    assert z._is_leaf is False
    assert w.requires_grad is True
    assert w._is_leaf is False
    
    