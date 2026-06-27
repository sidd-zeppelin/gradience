from gradience.tensor import Tensor

def test_tensor_creation():
    x = Tensor([1, 2, 3])
    
    assert x.shape == (3, )
    assert x.dtype.name == "float32"
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
