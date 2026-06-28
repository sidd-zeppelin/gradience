import os
import re

def inject_args(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # regex to find def test_xxx(): and replace with args if used in function
    def replacer(match):
        func_name = match.group(1)
        # find where this function ends
        start_idx = match.end()
        next_def = content.find('def test_', start_idx)
        if next_def == -1: next_def = len(content)
        func_body = content[start_idx:next_def]
        
        args = []
        if 'assert_eq(' in func_body: args.append('assert_eq')
        if 'assert_close(' in func_body: args.append('assert_close')
        
        if args:
            return f"def {func_name}({', '.join(args)}):"
        return match.group(0)
    
    new_content = re.sub(r'def (test_[a-zA-Z0-9_]+)\(\):', replacer, content)
    with open(filepath, 'w') as f:
        f.write(new_content)

for file in ['tests/test_tensor.py', 'tests/test_ops.py', 'tests/test_autograd.py', 'tests/test_gradcheck.py']:
    inject_args(file)

with open('tests/conftest.py', 'w') as f:
    f.write("""import pytest
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
""")
