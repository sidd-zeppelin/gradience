# Tensors: The Smart Data Containers

If you want to understand Gradience and neural networks, you have to start with the **Tensor**. 

In normal Python math, if you write `x = 5`, `x` is just a number. If you write `z = x + 2`, `z` becomes 7. But `z` has no idea where it came from. It forgot that it used to be `x + 2`.

In AI, forgetting where numbers come from is a big problem. We need to remember the history of our math so we can use the Chain Rule from calculus later on.

This is why Gradience uses Tensors instead of regular numbers.

## What is a Tensor?

Mathematically, a Tensor is just a grid of numbers. 
* A single number is a scalar (a 0D Tensor).
* A list of numbers is a vector (a 1D Tensor).
* A grid of numbers is a matrix (a 2D Tensor).
* A cube of numbers is a 3D Tensor.

In code, a Tensor is a Python class. You can find our version in `gradience/tensor.py`. 

## The Anatomy of a Tensor

When you create a Tensor, it holds five very important pieces of information. 

### 1. `_data`
This is where the actual numbers live. In Gradience, we use a tool called NumPy to store these numbers efficiently in the computer's memory. When you add two Tensors, you are really just adding their `_data` arrays together.

### 2. `_requires_grad`
This is a simple True or False switch. If you set this to True, you are telling the framework: "Hey! Pay attention to this Tensor! I am going to want its derivative later!" If it is False, the framework saves time and memory by ignoring it during the calculus phase.

### 3. `_grad`
This is an empty container at first. When the Autograd Engine finishes doing all the calculus, it will take the final derivative and store it right here in the `_grad` property. This way, you can easily look up the derivative of any variable.

### 4. `_grad_fn`
This stands for Gradient Function. This is how the Tensor remembers its history. If a Tensor was created by an addition operation, its `_grad_fn` will be a link to that specific addition operation. If the Tensor was created by you (and not by a math equation), this will be empty.

### 5. `_is_leaf`
Imagine a family tree. The people at the very top who have no parents in the record are called "leaves" (because family trees are drawn like real trees, but upside down). 
If you type `x = Tensor(5.0)`, `x` is a leaf because you created it directly. It has no math parents. 
If you type `z = x + 2`, `z` is NOT a leaf, because it was born from a math operation.

This distinction is important because Gradience usually only saves the final gradients into the leaf Tensors. 

## Let us Look at the Code

Here is how you might see a Tensor used:

```python
# x is a leaf. We want its gradient.
x = Tensor(2.0, requires_grad=True)

# y is a leaf. We want its gradient.
y = Tensor(3.0, requires_grad=True)

# z is NOT a leaf. It is born from multiplication. 
# Its _grad_fn will point to a Multiplication operation.
z = x * y
```

By wrapping our numbers inside this smart Tensor class, we have laid the foundation for Automatic Differentiation. The next step is to see how these Tensors link together to form a map. We cover that in the next section: **[The Computational Graph](autograd.md)**.
