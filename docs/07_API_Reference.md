# API Reference

This page is your cheat sheet for using Gradience in your own code! It lists all the tools and math operations you can use, explained simply.

## Creating Tensors

To do anything in Gradience, you first need to wrap your numbers inside a `Tensor`. 

```python
from gradience.tensor import Tensor

# Create a simple tensor (it will not track gradients)
x = Tensor([1.0, 2.0, 3.0])

# Create a tensor that tracks gradients (useful for AI learning)
w = Tensor(0.5, requires_grad=True)
```

## Tensor Properties

Once you have a Tensor, you can check its properties to see what is inside it.

*   `tensor.data`: Gives you the raw numbers (as a NumPy array).
*   `tensor.grad`: Gives you the calculated derivative after you run `.backward()`.
*   `tensor.shape`: Tells you the dimensions of your grid of numbers (like a 2x3 matrix).
*   `tensor.requires_grad`: A simple True/False telling you if this tensor is tracking gradients.
*   `tensor.is_leaf`: True if you created this tensor directly, False if it was born from a math operation.

## Tensor Utilities

*   `tensor.backward()`: Starts the Autograd Engine and calculates all the derivatives for the graph.
*   `tensor.zero_grad()`: Clears out the `grad` property so it is ready for a new learning loop.
*   `tensor.item()`: If your tensor is just a single number (a scalar), this gives you that number as a standard Python float.
*   `tensor.numpy()`: Gives you the raw NumPy array (same as `tensor.data`).
*   `tensor.clone()`: Creates an exact copy of the tensor.
*   `tensor.detach()`: Creates a copy of the tensor, but completely disconnects it from the Computational Graph.

## Basic Math Operations

You can use standard Python math symbols to do math with Tensors. 

```python
a = Tensor(10.0)
b = Tensor(2.0)

c = a + b    # Addition
d = a - b    # Subtraction
e = a * b    # Multiplication
f = a / b    # Division
g = a ** b   # Power (a to the power of b)
h = -a       # Negation (turns 10.0 into -10.0)
```

## Advanced Math & Trigonometry

We also support advanced math functions built directly into the Tensor.

*   `tensor.exp()`: Calculates the exponential (e^x).
*   `tensor.log(base)`: Calculates the logarithm.
*   `tensor.sqrt()`: Calculates the square root.
*   `tensor.sin()`: Calculates the sine.
*   `tensor.cos()`: Calculates the cosine.
*   `tensor.tan()`: Calculates the tangent.
*   `tensor.asin()`: Calculates the inverse sine (arcsine).
*   `tensor.acos()`: Calculates the inverse cosine (arccosine).
*   `tensor.atan()`: Calculates the inverse tangent (arctangent).

## Matrix Math and Reductions

When working with large grids of numbers, you will need these special operations.

*   `tensor.sum(axis, keepdims)`: Adds up all the numbers in the tensor. You can specify an axis to only add along rows or columns.
*   `tensor.mean(axis, keepdims)`: Finds the average of all the numbers in the tensor.
*   `tensor.matmul(other)`: Multiplies two matrices together. You can also use the `@` symbol in Python (like `a @ b`).

## Activation Functions (Neural Network Curves)

To make a Neural Network learn complex patterns, you need to use curved math operations.

*   `tensor.relu()`: If the number is negative, it turns it into 0. If it is positive, it leaves it alone.
*   `tensor.sigmoid()`: Squeezes any number into a small range between 0 and 1.
*   `tensor.tanh()`: Squeezes any number into a small range between -1 and 1.

## Neural Network Layers (gradience.nn)

Gradience provides high-level classes to build Neural Networks easily, abstracting away the raw tensor math.

*   `Module`: The base class for all neural network layers. Custom layers should inherit from this class and implement the `forward(self, x)` method. You can retrieve all trainable parameters in a module using `module.parameters()`.
*   `Parameter`: A special subclass of `Tensor`. Any tensor wrapped in `Parameter` is automatically registered as a learning weight inside a `Module` and will have `requires_grad=True` set by default.
*   `Linear(in_features, out_features, bias=True, *, weight_initializer=init.he_uniform, bias_initializer=init.zeros)`: A standard fully connected layer. It automatically creates the weight and bias `Parameter`s and performs the matrix multiplication (`x @ weight + bias`). You can optionally specify custom initializers.
*   `Sequential(*_modules)`: A sequential container. Modules will be added to it in the order they are passed in the constructor. Data passed to the forward pass of `Sequential` will be passed sequentially through each of the modules.
*   `ReLU()`, `Sigmoid()`, `Tanh()`: Neural Network layer wrappers for mathematical activation functions. They can be seamlessly inserted into a `Sequential` container.

## Loss Functions (gradience.nn.losses)

Gradience provides high-level objective functions to calculate the error of your Neural Network.

*   `MSELoss()`: Calculates the Mean Squared Error between the `prediction` and `target` tensors. Commonly used for regression tasks.

## Optimizers (gradience.optim)

Gradience provides optimization algorithms to update the parameters of your Neural Network based on computed gradients.

*   `Optimizer(parameters)`: Base class for all optimizers. Takes an iterable of `Parameter`s.
*   `SGD(parameters, lr=0.01)`: Implements Stochastic Gradient Descent. Calls `step()` to update parameters and `zero_grad()` to clear their gradients for the next pass.

## Initialization (gradience.nn.init)

Gradience provides several functions to properly initialize the weights of your Neural Networks to ensure healthy gradients.

*   `init.zeros(parameter)`: Fills the parameter with zeros.
*   `init.ones(parameter)`: Fills the parameter with ones.
*   `init.uniform(parameter, low, high)`: Fills with uniformly distributed random numbers.
*   `init.normal(parameter, mean, std)`: Fills with normally distributed random numbers.
*   `init.xavier_uniform(parameter)`: Xavier (Glorot) uniform initialization.
*   `init.xavier_normal(parameter)`: Xavier (Glorot) normal initialization.
*   `init.he_uniform(parameter)`: He (Kaiming) initialization.
