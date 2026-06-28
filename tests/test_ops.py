import numpy as np

from gradience.tensor import Tensor

def test_addition(assert_eq):
    
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    
    z = x + y
    assert_eq(
        z.data,
        [5, 7, 9]
    )
    
def test_multiplication(assert_eq):
    
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    
    z = x * y
    assert_eq(
        z.data,
        [4, 10, 18]
    )
    
def test_scalar_addition(assert_eq):
    x = Tensor([1, 2, 3])
    y = x + 3
    
    assert_eq(
        y.data,
        [4, 5, 6]
    )

def test_scalar_multiplication(assert_eq):
    x = Tensor([1, 2, 3])
    y = x * 5
    
    assert_eq(
        y.data,
        [5, 10, 15]
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
    
def test_subtraction(assert_eq):
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    
    z = x - y
    assert_eq(
        z.data,
        [-3, -3, -3]
    )
    
def test_scalar_subtraction(assert_eq):
    x = Tensor([1, 2, 3])
    y = x - 3
    
    assert_eq(
        y.data,
        [-2, -1, 0]
    )

def test_division(assert_eq):
    x = Tensor([6, 8, 10])
    y = Tensor([2, 4, 2])
    
    z = x / y
    assert_eq(
        z.data,
        [3, 2, 5]
    )

def test_scalar_division(assert_eq):
    x = Tensor([10, 20, 30])
    y = x / 2
    
    assert_eq(
        y.data,
        [5, 10, 15]
    )
    
    z = 12 / Tensor([2, 3, 4])
    assert_eq(
        z.data,
        [6, 4, 3]
    )

def test_power(assert_eq):
    x = Tensor([2.0, 3.0])
    y = Tensor([3.0, 2.0])
    
    z = x ** y
    assert_eq(
        z.data,
        [8.0, 9.0]
    )

def test_scalar_power(assert_eq):
    x = Tensor([2.0, 3.0])
    y = x ** 3.0
    
    assert_eq(
        y.data,
        [8.0, 27.0]
    )
    
    z = 2.0 ** Tensor([3.0, 4.0])
    assert_eq(
        z.data,
        [8.0, 16.0]
    )

    