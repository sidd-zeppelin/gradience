import numpy as np
from gradience.tensor import Tensor
from gradience.nn.layers.dropout import Dropout
from gradience.nn.layers.batchnorm import BatchNorm1d
from gradience.nn.layers.layernorm import LayerNorm

def test_dropout():
    dropout = Dropout(p=0.5)
    
    x = Tensor(np.ones((10, 10)))
    
    out_train = dropout(x)
    assert out_train.shape == (10, 10)
    zeros = np.sum(out_train.data == 0)
    assert 20 < zeros < 80 
    
    dropout.eval()
    out_eval = dropout(x)
    assert np.allclose(out_eval.data, np.ones((10, 10)))


def test_batchnorm1d():
    bn = BatchNorm1d(5)
    
    x = Tensor(np.random.randn(20, 5) * 2 + 1) 
    
    out = bn(x)
    
    np.testing.assert_allclose(out.data.mean(axis=0), 0, atol=1e-4)
    np.testing.assert_allclose(out.data.std(axis=0), 1, atol=1e-2)
    
    loss = out.sum()
    loss.backward()
    assert bn.weight.grad is not None
    assert bn.bias.grad is not None
    
    bn.eval()
    out_eval = bn(x)
    assert out_eval.shape == (20, 5)


def test_layernorm():
    ln = LayerNorm((5, 5))
    
    x = Tensor(np.random.randn(20, 5, 5) * 2 + 1)
    
    out = ln(x)
    
    sample_mean = out.data.mean(axis=(1, 2))
    sample_std = out.data.std(axis=(1, 2))
    
    np.testing.assert_allclose(sample_mean, 0, atol=1e-4)
    np.testing.assert_allclose(sample_std, 1, atol=1e-2)
    
    loss = out.sum()
    loss.backward()
    
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None
