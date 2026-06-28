from gradience.testing.gradcheck import (numerical_gradient,
                                         gradcheck)
from gradience.tensor import Tensor

def square(x):
    return x*x

def test_square_gradient():
    grad = numerical_gradient(
        square, 
        2.0
    )
    
    assert abs(grad - 4.0) < 1e-6
    
def cube(x):
    return x*x*x

def test_cube_gradient():
    grad = numerical_gradient(
        cube, 
        2.0
    )
    
    assert abs(grad - 12.0) < 1e-6
    
def test_tensor_square_gradient():
    x = Tensor([4.0], requires_grad=True)
    grad = numerical_gradient(
        square,
        4.0
    )
    
    y = square(x)
    y.backward()
    
    assert abs(x.grad - grad) < 1e-6
    
def test_tensor_cube_gradient():
    x = Tensor([4.0], requires_grad=True)
    grad = numerical_gradient(
        cube,
        4.0
    )
    
    y = cube(x)
    y.backward()
    
    assert abs(x.grad - grad) < 1e-6

def test_gradcheck_square():
    assert gradcheck(square, 2.0) is True

def test_gradcheck_cube():
    assert gradcheck(cube, 2.0) is True

def test_gradcheck_failure():
    import pytest
    from gradience.autograd.function import Function
    
    class BrokenOp(Function):
        @staticmethod
        def forward(ctx, x):
            return x * x
        @staticmethod
        def backward(ctx, grad_output):
            return (grad_output * 10.0,) # intentionally broken gradient
            
    def broken_square(x):
        return BrokenOp.apply(x)
        
    with pytest.raises(AssertionError):
        gradcheck(broken_square, 2.0)
    
