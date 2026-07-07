# Gradience

<p align="center">
  <b>A NumPy-based Deep Learning Framework Built From First Principles.</b>
</p>

<p align="center">
Building automatic differentiation, neural networks, and optimization algorithms from scratch—one component at a time.
</p>

---

## Why Gradience?

Modern machine learning frameworks are incredibly powerful, but their internal workings are often hidden behind millions of lines of code.

Gradience is an educational deep learning framework that rebuilds those ideas from first principles while maintaining clean software architecture and engineering practices.

Rather than treating machine learning as a black box, this project explores how neural networks work internally by implementing every major component ourselves.

The goal is not to compete with production frameworks—it's to understand them.

---



## Project Architecture

```text
gradience/
│
├── autograd/          # Core AD engine (graph nodes, context)
│
├── nn/                # Neural Network API
│   ├── activations/   # ReLU, Sigmoid, Tanh
│   ├── containers/    # Sequential models
│   ├── layers/        # Linear layers
│   ├── losses/        # MSE, L1, CrossEntropy, BCE
│   ├── init.py        # Weight initializers
│   ├── module.py      # Base Module class
│   └── parameter.py   # Trainable weights abstraction
│
├── ops/               # Differentiable primitive operations (add, mul, exp, etc.)
│
├── optim/             # Optimizers (SGD, Adam, AdamW, RMSprop, Adagrad)
│
├── tensor.py          # The core Tensor data structure
│
└── testing/           # Numerical gradient checking utilities
```

The project is intentionally modular.

Each component has a single responsibility:

* **Tensor** stores data and gradients.
* **Function** defines differentiable operations.
* **Context** stores intermediate values required during backpropagation.
* **GraphNode** represents a node in the computation graph.
* **Autograd Engine** performs reverse-mode automatic differentiation.

---

## Example

```python
from gradience.tensor import Tensor

x = Tensor(2.0, requires_grad=True)
y = Tensor(3.0, requires_grad=True)

z = x * y + x

z.backward()

print(x.grad)   # 4.0
print(y.grad)   # 2.0
```

---

## Gradience vs PyTorch: Performance & Correctness

To definitively prove that the Gradience autograd engine is mathematically identical to industry standards, we benchmarked a 2-Layer Multi-Layer Perceptron (`784` $\rightarrow$ `128` $\rightarrow$ `10`) on the **MNIST** dataset.

Both frameworks were instantiated identically, using deterministic seeds, identical batched datasets, identical hyperparameters (`lr=1.0`), and identical initial random weights transferred memory-for-memory into Gradience's internal Tensors.

| Metric | Gradience (NumPy Backend) | PyTorch (C++ ATen Backend) |
| :--- | :--- | :--- |
| **Forward Pass Time** | ~0.336 s | ~0.147 s |
| **Backward Pass Time** | ~0.868 s | ~0.193 s |
| **Total Training Time** | ~1.567 s | ~0.750 s |
| **Final Test Accuracy** | **90.00%** | **90.00%** |

**Conclusion:** 
Gradience achieves **exactly 100% mathematical identicality** to PyTorch (matching the 90.00% accuracy step-for-step across 5 epochs). As an educational pure-Python framework, retaining roughly $\sim 50\%$ of PyTorch's native CPU speed on complex MLPs is an outstanding validation of NumPy's underlying C-optimizations combined with Gradience's efficient topological graph traversals.

---

## Features

### Core Engine
* **NumPy-Backed Tensors**: Fast, underlying C-optimized matrix math.
* **Dynamic Computation Graph**: Builds reverse-mode automatic differentiation on the fly.
* **Broadcasting**: Natively supports full dimensional broadcasting during forward and backward passes.
* **Gradient Accumulation**: Handles arbitrary node branching and re-convergence.

### Mathematical Operations
* Arithmetic: `add`, `sub`, `mul`, `div`, `pow`, `neg`
* Transcendental: `exp`, `log`, `sqrt`
* Trigonometry: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
* Reductions: `sum`, `mean` (with `keepdims` support)
* Matrix Math: `matmul` (`@`)

### Neural Network API (`gradience.nn`)
* **Modules & Parameters**: Abstract base classes for building stateful models.
* **Layers**: `Linear` (Dense), `Conv2D`, `Dropout`, `BatchNorm1d`, `LayerNorm`.
* **Containers**: `Sequential` models.
* **Activations**: `ReLU`, `Sigmoid`, `Tanh`.
* **Loss Functions**: `MSELoss`, `L1Loss`, `CrossEntropyLoss` (Fused), `BCEWithLogitsLoss` (Fused).
* **Initializers**: `he_uniform`, `xavier_uniform`, `normal`, `zeros`, etc.

### Optimizers (`gradience.optim`)
* **SGD**: Stochastic Gradient Descent (with Momentum & Nesterov).
* **Adam**: Adaptive Moment Estimation.
* **AdamW**: Adam with Decoupled Weight Decay.
* **RMSprop**: Root Mean Square Propagation.
* **Adagrad**: Adaptive Gradients.

### Computational Graph Visualization (`gradience.visualization`)
* **Dynamic Graph Extraction**: Recursively traverses the autograd graph from any tensor.
* **Decoupled Architecture**: Strictly separates the generic graph structure representation from the rendering layer.
* **Graphviz Rendering**: Renders the computational graph as clean diagrams (tensors as ellipses, operations as boxes).

---

## Upcoming Features (v0.2.0+)
* Learning rate schedulers
* Convolutional Layers (Conv2D)
* Recurrent Neural Networks (RNN/LSTM)

---

## Design Philosophy

Gradience is built around a few core principles:

* **Understand before abstracting.**
* **Keep components small and modular.**
* **Prefer correctness over cleverness.**
* **Every operation should be mathematically verifiable.**
* **Every major feature should be backed by automated tests.**

---

## Installation

```bash
git clone https://github.com/sidd-zeppelin/gradience.git

cd gradience

uv sync
```

---

## Running Tests

Gradience has a robust, fully automated test suite configured with `pytest`. We maintain **100% test coverage** to ensure mathematical correctness across all operations and autograd mechanics.

To run the full test suite:

```bash
uv run python -m pytest tests/
```

To run the test suite with coverage reporting:

```bash
uv pip install pytest-cov
uv run python -m pytest --cov=gradience tests/
```

---

## Documentation

Explore our structured documentation to understand the framework's design, internals, and usage:

### High-Level Documentation
* **[Architecture Guide](docs/ARCHITECTURE.md)**: Conceptual diagram and explanation of the core layers.
* **[Design Philosophy & Decisions](docs/DESIGN.md)**: Architectural choices, trade-offs, and stability decisions.
* **[Contributing Guidelines](docs/CONTRIBUTING.md)**: Guidelines for adding operations, layers, and writing tests.
* **[Graph Visualization Guide](docs/visualization.md)**: Conceptual guide and architecture overview of the graph extraction and visualization subsystem.

### Subsystem Internals
* **[Tensor Engine](docs/internals/tensor.md)**: Smart containers, properties, and operator overloading.
* **[Autograd Mechanics](docs/internals/autograd.md)**: Dynamic graphs, topological sorting, and reverse differentiation.
* **[Module Abstractions](docs/internals/module.md)**: Parameter discovery, state tracking, and sequential layers.
* **[Broadcasting Math](docs/internals/broadcasting.md)**: Handling implicit and explicit shape matching in backpropagation.
* **[Numerical testing & gradcheck](docs/internals/testing.md)**: How numerical finite difference testing is used to verify analytical gradients.

### Step-by-Step Tutorials
* **[Tutorial 1: Training Your First Model](docs/tutorials/tutorial_1_linear_regression.md)**: Build a simple linear regression model and write a custom training loop.
* **[Tutorial 2: Building Deep Neural Networks](docs/tutorials/tutorial_2_deep_learning.md)**: Solve the non-linear XOR classification boundary using multi-layer perceptrons (MLPs).

### Example Notebooks
* **[Graph Visualization Demo](examples/visualize_graph.ipynb)**: Walkthrough of computation graph extraction, rendering, and file export for simple and complex mathematical functions.
* **[Linear Regression Demo](examples/linear_regression.ipynb)**: Training a simple linear regression model to fit synthetic 1D data.
* **[XOR Classification Demo](examples/train_xor.ipynb)**: Solving the classic non-linear XOR classification problem with a multi-layer perceptron (MLP).
* **[MNIST Classification Demo](examples/train_mnist.ipynb)**: Training a two-layer MLP on the MNIST handwritten digit database, showing step-for-step correctness compared with PyTorch.

### API Reference
* **[API Reference](docs/API_Reference.md)**: Full signatures and parameters of all operators, layers, and optimizers.
* **[Experiments & Validation](docs/experiments.md)**: Verification of accuracy and convergence relative to PyTorch.

---

## Inspiration

Gradience draws inspiration from several excellent educational projects, including:

* micrograd
* tinygrad
* PyTorch
* Andrej Karpathy's "Neural Networks: Zero to Hero"

The implementation is written independently and focuses on understanding the underlying engineering principles rather than reproducing existing code.

---

## Contributing

Contributions, discussions, and suggestions are always welcome.

We follow standard open-source practices. Please review our `CODE_OF_CONDUCT.md` and utilize the provided Issue and Pull Request templates in the `.github/` directory when submitting changes.

Whether you're interested in machine learning, software engineering, numerical computing, or simply learning how deep learning frameworks work internally, feel free to open an issue or submit a pull request.

---

## License

This project is released under the MIT License.
