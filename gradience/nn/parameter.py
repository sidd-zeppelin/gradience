from gradience.tensor import Tensor

class Parameter(Tensor):
    def __init__(self, data, requires_grad=True, dtype=None):
        super().__init__(data, requires_grad=requires_grad, dtype=dtype)
        
    def __repr__(self):
        return f"Parameter containing:\n{super().__repr__()}"
