import numpy as np

def _fill(parameter, values):
    parameter.data[...] = values
    return parameter

def _calculate_fan(parameter):
    shape = parameter.shape
    
    if len(shape) > 2:
        raise ValueError(
            "Fan in and fan out require a tensor with at least 2 dimensions."
        )
    
    fan_in = shape[0]
    fan_out = shape[1]
    
    return fan_in, fan_out

def constant(parameter, value):
    parameter.data[...] = value
    return parameter

def zeros(parameter):
    return constant(parameter, 0.0)

def ones(parameter):
    return constant(parameter, 1.0)

def uniform(parameter, low=1.0, high=1.0):
    return _fill(
        parameter, 
        np.random.uniform(
            low, 
            high,
            parameter.shape,
        ),
    )

def normal(parameter, mean=0.0, std=1.0):
    return _fill(
        parameter,
        np.random.normal(
            mean, 
            std,
            parameter.shape,
        ),
    )

def xavier_uniform(parameter):
    fan_in, fan_out = _calculate_fan(parameter)
    
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    
    return uniform(
        parameter, 
        -limit, 
        limit
    )
    
def xavier_normal(parameter):
    fan_in, fan_out = _calculate_fan(parameter)
    
    std = np.sqrt(2.0 / (fan_in + fan_out))
    
    return normal(
        parameter,
        mean=0.0,
        std=std,
    )    
    
def he_uniform(parameter):
    fan_in, _ = _calculate_fan(parameter)
    
    std = np.sqrt(2.0 / fan_in)
    
    return normal(
        parameter,
        mean = 0.0,
        std=std,
    )
    
