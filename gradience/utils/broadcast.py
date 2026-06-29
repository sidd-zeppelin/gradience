import numpy as np

def unbroadcast(grad: np.ndarray, original_shape: tuple) -> np.ndarray:
    if isinstance(grad, (float, int)):
        grad = np.array(grad)

    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)
        
    for i, dim in enumerate(original_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
            
    return grad
