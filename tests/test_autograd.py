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

def test_exponential_backward(assert_close):
    import numpy as np
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x.exp()
    y.backward()
    
    # d(e^x)/dx = e^x
    assert_close(x.grad, np.exp([1.0, 2.0]))

def test_log_backward(assert_close):
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x.log()
    y.backward()
    
    # d(ln x)/dx = 1/x
    assert_close(x.grad, [1.0, 0.5])
    
    x2 = Tensor([1.0, 2.0], requires_grad=True)
    z = x2.log(base=2)
    z.backward()
    
    import numpy as np
    assert_close(x2.grad, [1.0 / np.log(2), 0.5 / np.log(2)])

def test_sqrt_backward(assert_close):
    x = Tensor([1.0, 4.0], requires_grad=True)
    y = x.sqrt()
    y.backward()
    
    # d(sqrt(x))/dx = 1/(2*sqrt(x))
    assert_close(x.grad, [0.5, 0.25])

def test_sin_backward(assert_close):
    import numpy as np
    x = Tensor([0.0, np.pi/2], requires_grad=True)
    x.sin().backward()
    assert_close(x.grad, np.cos([0.0, np.pi/2]))

def test_cos_backward(assert_close):
    import numpy as np
    x = Tensor([0.0, np.pi/2], requires_grad=True)
    x.cos().backward()
    assert_close(x.grad, -np.sin([0.0, np.pi/2]))

def test_tan_backward(assert_close):
    import numpy as np
    x = Tensor([0.0, np.pi/4], requires_grad=True)
    x.tan().backward()
    assert_close(x.grad, 1.0 / (np.cos([0.0, np.pi/4]) ** 2))

def test_asin_backward(assert_close):
    import numpy as np
    x = Tensor([0.0, 0.5], requires_grad=True)
    x.asin().backward()
    assert_close(x.grad, 1.0 / np.sqrt(1.0 - np.array([0.0, 0.5])**2))

def test_acos_backward(assert_close):
    import numpy as np
    x = Tensor([0.0, 0.5], requires_grad=True)
    x.acos().backward()
    assert_close(x.grad, -1.0 / np.sqrt(1.0 - np.array([0.0, 0.5])**2))

def test_atan_backward(assert_close):
    import numpy as np
    x = Tensor([0.0, 1.0], requires_grad=True)
    x.atan().backward()
    assert_close(x.grad, 1.0 / (1.0 + np.array([0.0, 1.0])**2))

def test_sum_backward(assert_close):
    import numpy as np
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.sum()
    y.backward()
    assert_close(x.grad, np.ones((2, 2)))
    
    x2 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y2 = x2.sum(axis=0)
    y2.backward()
    assert_close(x2.grad, np.ones((2, 2)))

def test_mean_backward(assert_close):
    import numpy as np
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.mean()
    y.backward()
    assert_close(x.grad, np.ones((2, 2)) * 0.25)
    
    x2 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y2 = x2.mean(axis=1, keepdims=True)
    y2.backward()
    assert_close(x2.grad, np.ones((2, 2)) * 0.5)

    x3 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y3 = x3.mean(axis=(0, 1))
    y3.backward()
    assert_close(x3.grad, np.ones((2, 2)) * 0.25)

def test_matmul_backward(assert_close):
    import numpy as np
    
    # 2D @ 2D
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    
    z = x @ y
    z.backward()
    
    assert_close(x.grad, np.ones((2,2)) @ y.data.T)
    assert_close(y.grad, x.data.T @ np.ones((2,2)))
    
    # 1D @ 1D
    v1 = Tensor([1.0, 2.0], requires_grad=True)
    v2 = Tensor([3.0, 4.0], requires_grad=True)
    z2 = v1 @ v2
    z2.backward()
    
    assert_close(v1.grad, v2.data)
    assert_close(v2.grad, v1.data)
    
    # 1D @ 2D
    v3 = Tensor([1.0, 2.0], requires_grad=True)
    y3 = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    z3 = v3 @ y3
    z3.backward()
    
    assert_close(v3.grad, np.ones(2) @ y3.data.T)
    assert_close(y3.grad, np.outer(v3.data, np.ones(2)))
    
    # 2D @ 1D
    x4 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    v4 = Tensor([3.0, 4.0], requires_grad=True)
    z4 = x4 @ v4
    z4.backward()
    
    assert_close(x4.grad, np.outer(np.ones(2), v4.data))
    assert_close(v4.grad, x4.data.T @ np.ones(2))

def test_relu_backward(assert_close):
    import numpy as np
    x = Tensor([-2.0, 0.0, 2.0], requires_grad=True)
    x.relu().backward()
    assert_close(x.grad, np.array([0.0, 0.0, 1.0]))

def test_sigmoid_backward(assert_close):
    import numpy as np
    x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    x.sigmoid().backward()
    s = 1.0 / (1.0 + np.exp(-np.array([-1.0, 0.0, 1.0])))
    assert_close(x.grad, s * (1.0 - s))

def test_tanh_backward(assert_close):
    import numpy as np
    x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    x.tanh().backward()
    t = np.tanh([-1.0, 0.0, 1.0])
    assert_close(x.grad, 1.0 - t**2)