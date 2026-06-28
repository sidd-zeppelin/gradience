from gradience.tensor import Tensor

def test_tensor_creation():
    x = Tensor([1, 2, 3])
    
    assert x.shape == (3, )
    assert x.dtype.name == "int64"
    assert x.requires_grad is False
    assert x.grad is None
    assert x._is_leaf is True
    
def test_requires_grad():
    x = Tensor([1, 2, 3], requires_grad=True)
    
    assert x.requires_grad is True
    
def test_scalar_tensor():
    x = Tensor(5)
    
    assert x.shape == ()
    assert x.data.item() == 5.0

def test_item():
    x = Tensor(3.14)
    assert x.item() == 3.14

def test_numpy(assert_eq):
    x = Tensor([1, 2, 3])
    import numpy as np
    assert_eq(x.numpy(), [1, 2, 3])

def test_zero_grad():
    x = Tensor([1.0], requires_grad=True)
    x.grad = [2.0]
    x.zero_grad()
    assert x.grad is None

def test_clone(assert_eq):
    x = Tensor([1.0], requires_grad=True)
    y = x.clone()
    assert y.requires_grad is True
    assert y is not x
    import numpy as np
    assert_eq(x.data, y.data)

def test_detach(assert_eq):
    x = Tensor([1.0], requires_grad=True)
    y = x.detach()
    assert y.requires_grad is False
    assert y is not x
    import numpy as np
    assert_eq(x.data, y.data)

def test_tensor_dtype():
    import numpy as np
    x = Tensor([1, 2], dtype=np.float32)
    assert x.dtype.name == "float32"

def test_tensor_properties():
    x = Tensor([[1, 2], [3, 4]])
    assert x.is_leaf is True
    assert x.size == 4
    assert x.ndim == 2

def test_tensor_repr():
    x = Tensor(5)
    assert repr(x).startswith("Tensor(data=5,")

def test_rmul(assert_eq):
    x = Tensor([1, 2])
    y = 5 * x
    assert_eq(y.data, [5, 10])

def test_negation(assert_eq):
    x = Tensor([1, 2])
    y = -x
    assert_eq(y.data, [-1, -2])
