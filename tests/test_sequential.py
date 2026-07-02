import numpy as np

from gradience.nn.layers.linear import Linear
from gradience.nn.containers.sequential import Sequential
from gradience.tensor import Tensor


def test_sequential_initialization():
    seq = Sequential(
        Linear(3, 4),
        Linear(4, 2)
    )
    
    assert len(seq) == 2
    assert isinstance(seq[0], Linear)
    assert isinstance(seq[1], Linear)


def test_sequential_parameters():
    seq = Sequential(
        Linear(3, 4),
        Linear(4, 2)
    )
    
    params = seq.parameters()
    # 2 for first linear (weight, bias) + 2 for second linear = 4 parameters
    assert len(params) == 4
    
    # Assert gradients can be zeroed
    seq.zero_grad()


def test_sequential_forward():
    seq = Sequential(
        Linear(3, 4),
        Linear(4, 2)
    )
    
    x = Tensor(np.random.randn(5, 3))
    y = seq(x)
    
    assert y.shape == (5, 2)


def test_sequential_backward():
    seq = Sequential(
        Linear(3, 4),
        Linear(4, 2)
    )
    
    x = Tensor(np.random.randn(5, 3))
    y = seq(x)
    
    loss = y.sum()
    loss.backward()
    
    params = seq.parameters()
    for param in params:
        assert param.grad is not None


def test_sequential_train_eval():
    seq = Sequential(
        Linear(3, 4),
        Linear(4, 2)
    )
    
    seq.eval()
    assert seq.training is False
    assert seq[0].training is False
    assert seq[1].training is False
    
    seq.train()
    assert seq.training is True
    assert seq[0].training is True
    assert seq[1].training is True
