# API Reference

This document serves as the comprehensive, formal API specification for the **Gradience** framework. It details the arguments, return types, properties, and behaviors for every user-facing class and function.

---

## `gradience`

### `Tensor`
The core data structure in Gradience. It wraps a NumPy array and tracks operations to build a dynamic computation graph for reverse-mode automatic differentiation.

```python
from gradience import Tensor
```

**Constructor Signature:**
`Tensor(data: Union[float, int, list, np.ndarray], requires_grad: bool = False, dtype: Optional[np.dtype] = None)`

*   **Arguments:**
    *   `data`: The raw numerical data to wrap. Will be internally converted to a `numpy.ndarray`.
    *   `requires_grad`: Set to `True` to force the Autograd engine to track operations performed on this tensor.
    *   `dtype`: Force a specific NumPy data type (e.g., `np.float32`).

**Properties:**
*   `.data` (`np.ndarray`): Returns the underlying NumPy array. (Read-only property).
*   `.grad` (`Optional[np.ndarray]`): Returns the accumulated gradient array. Returns `None` if `.backward()` has not been called or if gradients were not tracked.
*   `.shape` (`tuple`): Returns the shape of the underlying array.
*   `.dtype` (`np.dtype`): Returns the data type of the underlying array.
*   `.requires_grad` (`bool`): Indicates if the tensor is tracking gradients.
*   `.is_leaf` (`bool`): Returns `True` if the tensor was created by the user (not as the result of a mathematical operation).
*   `.size` (`int`): The total number of elements in the tensor.
*   `.ndim` (`int`): The number of dimensions of the tensor.

**Methods:**
*   `.backward()`: Initiates the topological traversal of the computation graph, computing and accumulating gradients into the `.grad` attribute of all leaf tensors.
*   `.zero_grad()`: Clears the accumulated gradients (sets `.grad` to `None`).
*   `.item()` $\rightarrow$ `float`: Returns the value of this tensor as a standard Python scalar. Only valid for tensors with a single element.
*   `.numpy()` $\rightarrow$ `np.ndarray`: Returns a copy of the underlying NumPy array.
*   `.clone()` $\rightarrow$ `Tensor`: Returns a deep copy of the tensor, preserving the `requires_grad` state but detaching from the computation graph.
*   `.detach()` $\rightarrow$ `Tensor`: Returns a new tensor sharing the same `.data` array, but with `requires_grad=False` and disconnected from the computation graph.

---

## Tensor Mathematical Operations

Tensors natively support Python dunder-method operator overloading (e.g., `+`, `-`, `*`, `/`, `**`, `@`). When these operations are executed, they dynamically push new `GraphNode` objects to the computation graph.

### Arithmetic
*   `__add__(other)`: Element-wise addition (`a + b`). Supports broadcasting.
*   `__sub__(other)`: Element-wise subtraction (`a - b`). Supports broadcasting.
*   `__mul__(other)`: Element-wise multiplication (`a * b`). Supports broadcasting.
*   `__truediv__(other)`: Element-wise division (`a / b`). Supports broadcasting.
*   `__pow__(other)`: Element-wise power (`a ** b`).
*   `__neg__()`: Element-wise negation (`-a`).

### Reductions
*   `.sum(axis=None, keepdims=False)` $\rightarrow$ `Tensor`: Sums elements along a specified axis.
*   `.mean(axis=None, keepdims=False)` $\rightarrow$ `Tensor`: Computes the arithmetic mean along a specified axis.

### Matrix Operations
*   `.matmul(other)` $\rightarrow$ `Tensor`: Performs matrix multiplication. Equivalently accessed via the `@` operator (`a @ b`).

### Non-Linear & Transcendental 
*   `.exp()` $\rightarrow$ `Tensor`: Computes the element-wise exponential $e^x$.
*   `.log(base=np.e)` $\rightarrow$ `Tensor`: Computes the element-wise logarithm to the specified base.
*   `.sqrt()` $\rightarrow$ `Tensor`: Computes the element-wise square root.
*   `.sin()` $\rightarrow$ `Tensor`: Computes the element-wise sine.
*   `.cos()` $\rightarrow$ `Tensor`: Computes the element-wise cosine.
*   `.tan()` $\rightarrow$ `Tensor`: Computes the element-wise tangent.
*   `.asin()` $\rightarrow$ `Tensor`: Computes the element-wise inverse sine.
*   `.acos()` $\rightarrow$ `Tensor`: Computes the element-wise inverse cosine.
*   `.atan()` $\rightarrow$ `Tensor`: Computes the element-wise inverse tangent.
*   `.relu()` $\rightarrow$ `Tensor`: Applies the Rectified Linear Unit activation.
*   `.sigmoid()` $\rightarrow$ `Tensor`: Applies the Sigmoid activation.
*   `.tanh()` $\rightarrow$ `Tensor`: Applies the Hyperbolic Tangent activation.

---

## `gradience.nn` (Neural Network API)

This subpackage contains higher-level abstractions for defining layered neural networks.

```python
from gradience.nn import Parameter, Module, Linear, Sequential
```

### `Parameter(Tensor)`
A direct subclass of `Tensor`. 
It serves as a marker for the `Module` base class. When a `Module` scans its attributes, any `Parameter` objects are automatically registered as trainable weights.
*   **Default Behavior:** Forces `requires_grad=True`.

### `Module`
The base class for all neural network layers and containers. Custom architectures should inherit from this class.
*   **Methods:**
    *   `.forward(*args, **kwargs)`: Abstract method. Must be overridden to define the forward pass logic.
    *   `.parameters()` $\rightarrow$ `List[Parameter]`: Recursively scans the module and all child modules, returning a flat list of all registered `Parameter`s.
    *   `.train()`: Sets the module (and all child modules) into training mode.
    *   `.eval()`: Sets the module (and all child modules) into evaluation mode.
    *   `__call__(*args, **kwargs)`: Invokes the `.forward()` method natively.

### `Linear(Module)`
A fully-connected dense layer. Applies a linear transformation $y = xW^T + b$.
*   **Constructor:** `Linear(in_features: int, out_features: int, bias: bool = True, *, weight_initializer=init.he_uniform, bias_initializer=init.zeros)`
*   **Attributes:**
    *   `.weight` (`Parameter`): The learnable weight matrix of shape `(out_features, in_features)`.
    *   `.bias` (`Optional[Parameter]`): The learnable bias vector of shape `(out_features,)`.

### `Sequential(Module)`
A sequential container for chaining modules.
*   **Constructor:** `Sequential(*_modules: Module)`
*   **Behavior:** During the forward pass, the output of each module is passed identically as the input to the next.

### `ReLU(Module)`, `Sigmoid(Module)`, `Tanh(Module)`
Object-oriented wrappers for the underlying Tensor activation functions, allowing them to be inserted seamlessly into `Sequential` containers.

---

## `gradience.nn.losses`

```python
from gradience.nn.losses import MSELoss, L1Loss, CrossEntropyLoss, BCEWithLogitsLoss
```

### `MSELoss(Module)`
Calculates the Mean Squared Error between the predicted tensor and the target tensor.
*   **Constructor:** `MSELoss()`
*   **Forward Call:** `loss_fn(prediction: Tensor, target: Tensor)` $\rightarrow$ `Tensor`
*   **Behavior:** Returns a scalar Tensor representing the average of the squared differences across all dimensions and batch elements.

### `L1Loss(Module)`
Calculates the Mean Absolute Error between the predicted tensor and the target tensor.
*   **Constructor:** `L1Loss()`
*   **Forward Call:** `loss_fn(prediction: Tensor, target: Tensor)` $\rightarrow$ `Tensor`

### `CrossEntropyLoss(Module)`
Calculates the Cross Entropy Loss between predictions and targets.
*   **Constructor:** `CrossEntropyLoss()`
*   **Behavior:** This function expects *unnormalized logits* as inputs. It dynamically fuses Log-Softmax and Negative Log-Likelihood Loss internally for maximum numerical stability. Targets can be passed as raw integer class indices (e.g. `[0, 1]`) or one-hot encoded vectors.

### `BCEWithLogitsLoss(Module)`
Calculates the Binary Cross Entropy Loss between target and predictions.
*   **Constructor:** `BCEWithLogitsLoss()`
*   **Behavior:** Expects *unnormalized logits* as predictions and automatically applies a Sigmoid activation internally utilizing the log-sum-exp trick for numerical stability.

---

## `gradience.optim`

```python
from gradience.optim import SGD, Adam, AdamW, RMSprop, Adagrad
```

### `Optimizer`
The base class for all gradient descent algorithms.
*   **Constructor:** `Optimizer(parameters: Iterable[Parameter])`
*   **Methods:**
    *   `.zero_grad()`: Iterates over all registered parameters and calls `.zero_grad()` on them, clearing their gradients.
    *   `.step()`: Abstract method to apply the gradient update.

### `SGD(Optimizer)`
Implements standard Stochastic Gradient Descent with optional momentum and weight decay.
*   **Constructor:** `SGD(parameters: Iterable[Parameter], lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0, nesterov: bool = False)`

### `Adam(Optimizer)`
Implements the Adam optimization algorithm.
*   **Constructor:** `Adam(parameters: Iterable[Parameter], lr: float = 0.001, betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0)`

### `AdamW(Optimizer)`
Implements the AdamW optimization algorithm (Adam with decoupled weight decay).
*   **Constructor:** `AdamW(parameters: Iterable[Parameter], lr: float = 0.001, betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01)`

### `RMSprop(Optimizer)`
Implements the RMSprop optimization algorithm.
*   **Constructor:** `RMSprop(parameters: Iterable[Parameter], lr: float = 0.01, alpha: float = 0.99, eps: float = 1e-8, weight_decay: float = 0.0, momentum: float = 0.0)`

### `Adagrad(Optimizer)`
Implements the Adagrad optimization algorithm.
*   **Constructor:** `Adagrad(parameters: Iterable[Parameter], lr: float = 0.01, eps: float = 1e-10, weight_decay: float = 0.0)`

---

## `gradience.nn.init`

Utility functions to fill `Parameter` tensors with specific statistical distributions. These manipulate the `._data` array directly in-place.

*   `init.zeros(parameter: Parameter)`: Fills with `0.0`.
*   `init.ones(parameter: Parameter)`: Fills with `1.0`.
*   `init.uniform(parameter: Parameter, low: float, high: float)`: Fills with a uniform distribution $U(low, high)$.
*   `init.normal(parameter: Parameter, mean: float, std: float)`: Fills with a normal distribution $\mathcal{N}(\mu, \sigma^2)$.
*   `init.xavier_uniform(parameter: Parameter)`: Glorot Uniform initialization.
*   `init.xavier_normal(parameter: Parameter)`: Glorot Normal initialization.
*   `init.he_uniform(parameter: Parameter)`: Kaiming Uniform initialization (optimal for ReLU networks).
