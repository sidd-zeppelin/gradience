import pytest

from gradience.tensor import Tensor

@pytest.fixture
def tensors():
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    
    return x, y

...