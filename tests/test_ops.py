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

def test_exponential(assert_close):
    import numpy as np
    x = Tensor([1.0, 2.0])
    y = x.exp()
    assert_close(y.data, np.exp([1.0, 2.0]))

def test_log(assert_close):
    import numpy as np
    x = Tensor([1.0, np.e])
    y = x.log()
    assert_close(y.data, np.log([1.0, np.e]))
    
    z = x.log(base=2)
    assert_close(z.data, np.log([1.0, np.e]) / np.log(2))

def test_sqrt(assert_close):
    import numpy as np
    x = Tensor([1.0, 4.0])
    y = x.sqrt()
    assert_close(y.data, np.sqrt([1.0, 4.0]))

def test_sum(assert_close):
    import numpy as np
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert_close(x.sum().data, np.sum([[1.0, 2.0], [3.0, 4.0]]))
    assert_close(x.sum(axis=0).data, np.sum([[1.0, 2.0], [3.0, 4.0]], axis=0))
    assert_close(x.sum(axis=1, keepdims=True).data, np.sum([[1.0, 2.0], [3.0, 4.0]], axis=1, keepdims=True))

def test_mean(assert_close):
    import numpy as np
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert_close(x.mean().data, np.mean([[1.0, 2.0], [3.0, 4.0]]))
    assert_close(x.mean(axis=0).data, np.mean([[1.0, 2.0], [3.0, 4.0]], axis=0))
    assert_close(x.mean(axis=1, keepdims=True).data, np.mean([[1.0, 2.0], [3.0, 4.0]], axis=1, keepdims=True))

def test_sin(assert_close):
    import numpy as np
    x = Tensor([0.0, np.pi/2])
    assert_close(x.sin().data, np.sin([0.0, np.pi/2]))

def test_matmul(assert_close):
    import numpy as np
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = Tensor([[5.0, 6.0], [7.0, 8.0]])
    
    assert_close((x @ y).data, np.matmul(x.data, y.data))
    assert_close(x.matmul(y).data, np.matmul(x.data, y.data))
    
    # test 1D dot product
    v1 = Tensor([1.0, 2.0])
    v2 = Tensor([3.0, 4.0])
    assert_close((v1 @ v2).data, np.matmul(v1.data, v2.data))
    
    # test 1D @ 2D
    assert_close((v1 @ y).data, np.matmul(v1.data, y.data))
    
    # test 2D @ 1D
    assert_close((x @ v2).data, np.matmul(x.data, v2.data))
    
    # test __rmatmul__
    assert_close((x.data @ y).data, np.matmul(x.data, y.data))

def test_relu(assert_close):
    import numpy as np
    x = Tensor([-2.0, 0.0, 2.0])
    y = x.relu()
    assert_close(y.data, np.array([0.0, 0.0, 2.0]))

def test_sigmoid(assert_close):
    import numpy as np
    x = Tensor([-1.0, 0.0, 1.0])
    y = x.sigmoid()
    assert_close(y.data, 1.0 / (1.0 + np.exp(-np.array([-1.0, 0.0, 1.0]))))

def test_tanh(assert_close):
    import numpy as np
    x = Tensor([-1.0, 0.0, 1.0])
    y = x.tanh()
    assert_close(y.data, np.tanh([-1.0, 0.0, 1.0]))

def test_cos(assert_close):
    import numpy as np
    x = Tensor([0.0, np.pi/2])
    assert_close(x.cos().data, np.cos([0.0, np.pi/2]))

def test_tan(assert_close):
    import numpy as np
    x = Tensor([0.0, np.pi/4])
    assert_close(x.tan().data, np.tan([0.0, np.pi/4]))

def test_asin(assert_close):
    import numpy as np
    x = Tensor([0.0, 1.0])
    assert_close(x.asin().data, np.arcsin([0.0, 1.0]))

def test_acos(assert_close):
    import numpy as np
    x = Tensor([0.0, 1.0])
    assert_close(x.acos().data, np.arccos([0.0, 1.0]))

def test_atan(assert_close):
    import numpy as np
    x = Tensor([0.0, 1.0])
    assert_close(x.atan().data, np.arctan([0.0, 1.0]))

    