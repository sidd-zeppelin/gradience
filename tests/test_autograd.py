import numpy as np

from gradience.tensor import Tensor

def test_add_backward():
    x = Tensor([1.0], requires_grad=True)
    y = Tensor([4.0], requires_grad=True)
    
    z = x + y
    z.backward()
    
    np.testing.assert_array_equal(x.grad, [1.0])
    np.testing.assert_array_equal(y.grad, [1.0])
    
def test_multiply_backward():
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)
    
    z = x*y
    z.backward()
    
    np.testing.assert_array_equal(x.grad, [3.0])
    np.testing.assert_array_equal(y.grad, [2.0])
    
def test_chain_rule():
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)
    
    m = x * y
    n = m + x
    
    n.backward()
    np.testing.assert_array_equal(x.grad, [4.0])
    np.testing.assert_array_equal(y.grad, [2.0])
    
def test_gradient_accumulation():
    x = Tensor([2.0], requires_grad=True)
    
    y = x*x
    
    y.backward()
    np.testing.assert_array_equal(x.grad, [4.0])
        