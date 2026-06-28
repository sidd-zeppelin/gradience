import numpy as np

from gradience.tensor import Tensor

def test_add_backward(assert_eq):
    x = Tensor([1.0], requires_grad=True)
    y = Tensor([4.0], requires_grad=True)
    
    z = x + y
    z.backward()
    
    assert_eq(x.grad, [1.0])
    assert_eq(y.grad, [1.0])
    
def test_multiply_backward(assert_eq):
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)
    
    z = x*y
    z.backward()
    
    assert_eq(x.grad, [3.0])
    assert_eq(y.grad, [2.0])
    
def test_chain_rule(assert_eq):
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)
    
    m = x * y
    n = m + x
    
    n.backward()
    assert_eq(x.grad, [4.0])
    assert_eq(y.grad, [2.0])
    
def test_gradient_accumulation(assert_eq):
    x = Tensor([2.0], requires_grad=True)
    
    y = x*x
    
    y.backward()
    assert_eq(x.grad, [4.0])
    
def test_subtraction_backward(assert_eq):
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)
    
    z = x - y
    z.backward()
    
    assert_eq(x.grad, [1.0])
    assert_eq(y.grad, [1.0])
            
def test_scalar_subtraction_backward(assert_eq):
    x = Tensor([5.0], requires_grad=True)
    
    y = x - 5
    y.backward()
    
    assert_eq(x.grad, [1.0])

def test_division_backward(assert_eq):
    x = Tensor([6.0], requires_grad=True)
    y = Tensor([2.0], requires_grad=True)
    
    z = x / y
    z.backward()
    
    assert_eq(x.grad, [0.5])
    assert_eq(y.grad, [-1.5])

def test_scalar_division_backward(assert_eq):
    x = Tensor([10.0], requires_grad=True)
    
    y = x / 2
    y.backward()
    
    assert_eq(x.grad, [0.5])
    
    x2 = Tensor([8.0], requires_grad=True)
    z = 8 / x2
    z.backward()
    
    assert_eq(x2.grad, [-0.125])

def test_power_backward(assert_close):
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)
    
    z = x ** y
    z.backward()
    
    assert_close(x.grad, [12.0])

    assert_close(y.grad, [8.0 * np.log(2.0)])

def test_scalar_power_backward(assert_close):
    x = Tensor([3.0], requires_grad=True)
    
    y = x ** 2.0
    y.backward()
    
    assert_close(x.grad, [6.0])
    
    x2 = Tensor([2.0], requires_grad=True)
    z = 3.0 ** x2
    z.backward()
    
    assert_close(x2.grad, [9.0 * np.log(3.0)])

def test_negation_backward(assert_eq):
    x = Tensor([2.0, -3.0], requires_grad=True)
    y = -x
    y.backward()
    assert_eq(x.grad, [-1.0, -1.0])