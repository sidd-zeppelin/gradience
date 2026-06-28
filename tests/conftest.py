import pytest
import numpy as np
from gradience.tensor import Tensor

@pytest.fixture
def assert_eq():
    def _assert(actual, expected):
        if not isinstance(expected, np.ndarray) and not isinstance(expected, Tensor):
            expected = np.array(expected, dtype=actual.dtype if hasattr(actual, 'dtype') else np.float32)
        np.testing.assert_array_equal(actual, expected)
    return _assert

@pytest.fixture
def assert_close():
    def _assert(actual, expected, rtol=1e-5, atol=1e-8):
        if not isinstance(expected, np.ndarray) and not isinstance(expected, Tensor):
            expected = np.array(expected, dtype=actual.dtype if hasattr(actual, 'dtype') else np.float32)
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    return _assert

@pytest.fixture
def tensors():
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    return x, y
