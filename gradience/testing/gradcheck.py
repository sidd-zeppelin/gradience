import numpy as np

from gradience.tensor import Tensor
_GRADCHECK_DTYPE = np.float64

def numerical_gradient(
    function, 
    x, 
    eps=1e-5,
):
    
    forward = function(x + eps)
    backward = function(x - eps)
 
    return (forward - backward) / (2 * eps)

def gradcheck(
    function,
    *inputs,
    eps=1e-5,
    atol=1e-6,
):

    tensors = tuple(
        Tensor(
            value,
            requires_grad=True,
            dtype=_GRADCHECK_DTYPE,
        )
        for value in inputs
    )

    output = function(*tensors)
    
    output.backward()
    analytical = tuple(
        tensor.grad.item()
        for tensor in tensors
    )
    numerical = []
    for index, _ in enumerate(inputs):
        def wrapped(value):
            pertubed_inputs = list(inputs)
            pertubed_inputs[index] = value
            tensors = tuple(
                Tensor(
                    value,
                    dtype=_GRADCHECK_DTYPE,
                )
                for value in pertubed_inputs
            )
            return function(*tensors).data.item()
        
        gradient = numerical_gradient(
            wrapped,
            inputs[index],
            eps,
        )
        numerical.append(gradient)
    
    numerical = tuple(numerical)
    differences = tuple(
        abs(a - n)
        for a, n in zip(
            analytical,
            numerical,
        )
    )
    
    if any(
        difference > atol
        for difference in differences
    ):
        raise AssertionError(
            (
                f"Gradient check failed!\n\n"
                f"Function   : {function.__name__}\n"
                f"Input      : {inputs}\n\n"
                f"Analytical : {analytical}\n"
                f"Numerical  : {numerical}\n"
                f"Difference : {differences}\n"
                f"Tolerance  : {atol}"
            )
        )
    return True
    