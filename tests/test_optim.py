import numpy as np
from gradience.tensor import Tensor
from gradience.optim import SGD, Adam, AdamW, RMSprop, Adagrad
from gradience.nn.parameter import Parameter

def test_sgd_step():
    p = Parameter(np.array([1.0, 2.0]))
    opt = SGD([p], lr=0.1)
    
    # Fake gradient
    p.grad = np.array([0.5, 0.5])
    opt.step()
    
    np.testing.assert_allclose(p.data, np.array([0.95, 1.95]))
    
def test_sgd_momentum():
    p = Parameter(np.array([1.0]))
    opt = SGD([p], lr=0.1, momentum=0.9)
    
    p.grad = np.array([1.0])
    opt.step()
    # velocity = 1.0, p -= 0.1 * 1.0 => 0.9
    np.testing.assert_allclose(p.data, np.array([0.9]))
    
    opt.step()
    # velocity = 0.9 * 1.0 + 1.0 = 1.9
    # p = 0.9 - 0.1 * 1.9 = 0.71
    np.testing.assert_allclose(p.data, np.array([0.71]))

def test_adam_step():
    p = Parameter(np.array([1.0, 2.0]))
    opt = Adam([p], lr=0.1)
    
    p.grad = np.array([0.5, 0.5])
    opt.step()
    
    # Just check it ran without error and updated
    assert not np.array_equal(p.data, np.array([1.0, 2.0]))

def test_adamw_step():
    p = Parameter(np.array([1.0, 2.0]))
    opt = AdamW([p], lr=0.1, weight_decay=0.01)
    
    p.grad = np.array([0.5, 0.5])
    opt.step()
    
    assert not np.array_equal(p.data, np.array([1.0, 2.0]))
    
def test_rmsprop_step():
    p = Parameter(np.array([1.0, 2.0]))
    opt = RMSprop([p], lr=0.1)
    
    p.grad = np.array([0.5, 0.5])
    opt.step()
    
    assert not np.array_equal(p.data, np.array([1.0, 2.0]))

def test_adagrad_step():
    p = Parameter(np.array([1.0, 2.0]))
    opt = Adagrad([p], lr=0.1)
    
    p.grad = np.array([0.5, 0.5])
    opt.step()
    
    assert not np.array_equal(p.data, np.array([1.0, 2.0]))
