import numpy as np


class Tensor:
    
    
    # Tensor initialization
    def __init__(self, data, requires_grad=False):
        self._data = np.asarray(data, dtype=np.float32)
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
            f"requires_grad={self.requires_grad})"
        )
    
    
    @staticmethod
    def _as_tensor(value):
        if isinstance(value, Tensor):
            return value
        return Tensor(value)
    
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