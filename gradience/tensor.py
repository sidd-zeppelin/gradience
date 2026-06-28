from numpy._core import function_base
import numpy as np
from gradience.autograd.autograd_engine import AutogradEngine


class Tensor:
    
    __array_priority__ = 1000
    __array_ufunc__ = None
    
    # Tensor initialization
    def __init__(self, data, requires_grad=False, dtype=None):
        if dtype is None:
            self._data = np.asarray(data)
        else:
            self._data = np.asarray(data, dtype=dtype)
        self._grad = None
        self._requires_grad = requires_grad
        
        self._grad_fn = None
        self._is_leaf = True
    
    #class method to generate tensor for operation
    @classmethod
    def _from_operation(
        cls,
        data,
        *,
        requires_grad,
        grad_fn=None,
    ):
        tensor = cls(
            data,
            requires_grad=requires_grad
        )
        tensor._is_leaf = False
        tensor._grad_fn = grad_fn
        
        return tensor
    
    #tensor properties
    @property
    def data(self):
        return self._data
    
    @property
    def grad(self):
        return self._grad
    
    @grad.setter
    def grad(self, value):
        self._grad = value
        
    @property
    def shape(self):
        return self._data.shape
        
    @property
    def dtype(self):
        return self._data.dtype
    
    @property
    def requires_grad(self):
        return self._requires_grad
    
    @property
    def is_leaf(self):
        return self._is_leaf
    
    @property
    def grad_fn(self):
        return self._grad_fn
    
    @property
    def size(self):
        return self._data.size
    
    @property
    def ndim(self):
        return self._data.ndim
    
    def __repr__(self):
        return (
            f"Tensor(data={self.data}, "
            f"shape={self.shape}, "
            f"dtype={self.dtype}, "
            f"requires_grad={self.requires_grad}, "
            f"is_leaf={self._is_leaf})\n"
        )
    
    @staticmethod
    def _as_tensor(value):
        if isinstance(value, Tensor):
            return value
        return Tensor(value)
    
    def backward(self):
        AutogradEngine.backward(self)

    def item(self):
        return self._data.item()

    def numpy(self):
        return self._data

    def zero_grad(self):
        self.grad = None    
    
    def clone(self):
        return Tensor(
            self.data.copy(),
            requires_grad=self.requires_grad    
        )
    
    def detach(self):
        tensor = Tensor(
            self.data,
            requires_grad = False
        )
        return tensor

    # operation definitions
    def __add__(self, other):
        from gradience.ops.add import AddOp
        
        other = Tensor._as_tensor(other)
        return AddOp.apply(self, other)
        
    def __mul__(self, other):
        from gradience.ops.multiply import MultOp
        
        other = Tensor._as_tensor(other)
        return MultOp.apply(self, other)
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __sub__(self, other):
        from gradience.ops.subtract import SubtractOp
        
        other = Tensor._as_tensor(other)
        return SubtractOp.apply(self, other)

    def __truediv__(self, other):
        from gradience.ops.division import DivisionOp
        
        other = Tensor._as_tensor(other)
        return DivisionOp.apply(self, other)

    def __rtruediv__(self, other):
        from gradience.ops.division import DivisionOp
        
        other = Tensor._as_tensor(other)
        return DivisionOp.apply(other, self)
    
    def __neg__(self):
        from gradience.ops.negation import NegationOp
        return NegationOp.apply(self)   

    def __pow__(self, other):
        from gradience.ops.power import PowerOp
        
        other = Tensor._as_tensor(other)
        return PowerOp.apply(self, other)

    def __rpow__(self, other):
        from gradience.ops.power import PowerOp
        
        other = Tensor._as_tensor(other)
        return PowerOp.apply(other, self)

    def exp(self):
        from gradience.ops.exponential import ExpOp
        return ExpOp.apply(self)

    def log(self, base=None):
        import numpy as np
        if base is None:
            base = np.e
        from gradience.ops.logarithm import LogOp
        return LogOp.apply(self, base=base)
        
    def sqrt(self):
        from gradience.ops.sqrt import SqrtOp
        return SqrtOp.apply(self)

    def sin(self):
        from gradience.ops.sin import SinOp
        return SinOp.apply(self)

    def cos(self):
        from gradience.ops.cos import CosOp
        return CosOp.apply(self)

    def tan(self):
        from gradience.ops.tan import TanOp
        return TanOp.apply(self)

    def asin(self):
        from gradience.ops.asin import ASinOp
        return ASinOp.apply(self)

    def acos(self):
        from gradience.ops.acos import ACosOp
        return ACosOp.apply(self)

    def atan(self):
        from gradience.ops.atan import ATanOp
        return ATanOp.apply(self)

    def sum(self, axis=None, keepdims=False):
        from gradience.ops.sum import SumOp
        return SumOp.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        from gradience.ops.mean import MeanOp
        return MeanOp.apply(self, axis=axis, keepdims=keepdims)

    def __matmul__(self, other):
        from gradience.ops.matmul import MatMulOp
        other = Tensor._as_tensor(other)
        return MatMulOp.apply(self, other)

    def __rmatmul__(self, other):
        from gradience.ops.matmul import MatMulOp
        other = Tensor._as_tensor(other)
        return MatMulOp.apply(other, self)

    def matmul(self, other):
        return self.__matmul__(other)

    def relu(self):
        from gradience.ops.relu import ReLUOp
        return ReLUOp.apply(self)

    def sigmoid(self):
        from gradience.ops.sigmoid import SigmoidOp
        return SigmoidOp.apply(self)

    def tanh(self):
        from gradience.ops.tanh import TanhOp
        return TanhOp.apply(self)