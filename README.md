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

## Current Progress

### Core Tensor Engine

* NumPy-backed tensors
* Gradient storage
* Automatic computation graph construction
* Dynamic graph execution
* Operator overloading

### Reverse-Mode Automatic Differentiation

* Reverse topological traversal
* Chain rule implementation
* Gradient accumulation
* Context-based operation execution
* Dynamic computation graph

### Supported Operations

* Addition
* Multiplication
* Subtraction
* Division
* Power
* Negation
* Exponential
* Logarithm
* Square Root
* Trigonometric (sin, cos, tan, asin, acos, atan)
* Reductions (sum, mean)
* Matrix Multiplication (matmul)

### Testing Infrastructure

* Unit tests
* Autograd tests
* Numerical gradient checking
* Multi-input gradient verification

### Neural Network API

* `Module` base class for stateful tracking
* `Parameter` abstractions for dynamic weight discovery
* `Linear` (Dense) layers with custom initializers
* `Sequential` containers with iterable Python indexing

---

## Project Architecture

```text
gradience/
│
├── autograd/
│   ├── autograd_engine.py
│   ├── context.py
│   ├── function.py
│   └── graph_node.py
│
├── nn/
│   ├── activations/
│   ├── containers/
│   ├── layers/
│   │   └── linear.py
│   ├── losses/
│   ├── init.py
│   ├── module.py
│   └── parameter.py
│
├── ops/
│   ├── add.py
│   ├── multiply.py
│   ├── subtract.py
│   ├── division.py
│   ├── power.py
│   ├── negation.py
│   ├── exponential.py
│   ├── logarithm.py
│   ├── sqrt.py
│   ├── sin.py
│   ├── cos.py
│   ├── tan.py
│   ├── asin.py
│   ├── acos.py
│   ├── atan.py
│   ├── sum.py
│   ├── mean.py
│   ├── matmul.py
│   ├── relu.py
│   ├── sigmoid.py
│   └── tanh.py
│
├── optim/
│
├── tensor.py
│
└── testing/
    └── gradcheck.py
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

## Roadmap

### Phase 1 — Core Autograd Engine

* [x] Tensor implementation
* [x] Dynamic computation graph
* [x] Reverse-mode automatic differentiation
* [x] Gradient accumulation
* [x] Numerical gradient checking

### Phase 2 — Mathematical Operations

* [x] Subtraction
* [x] Division
* [x] Power
* [x] Negation
* [x] Exponential
* [x] Logarithm
* [x] Square Root
* [x] Trigonometric Functions
* [x] Reductions (sum, mean)
* [x] Matrix multiplication
* [x] Broadcasting

### Phase 3 — Neural Network API

* [x] Activations (ReLU, Sigmoid, Tanh)
* [x] Parameter abstraction
* [x] Base Module class
* [x] Linear/Dense Layer
* [x] Sequential container
* [x] Activation functions
* [x] Loss functions

### Phase 4 — Training

* [x] Optimizers
* [x] SGD
* [ ] Adam
* [ ] Learning rate schedulers
* [x] End-to-end neural network training

---

## Design Philosophy

Gradience is built around a few core principles:

* **Understand before abstracting.**
* **Keep components small and modular.**
* **Prefer correctness over cleverness.**
* **Every operation should be mathematically verifiable.**
* **Every major feature should be backed by automated tests.**

---

## Development Workflow

Every new differentiable operation follows the same process:

```text
Implement Forward
        │
        ▼
Implement Backward
        │
        ▼
Unit Tests
        │
        ▼
Gradient Check
        │
        ▼
Commit
```

This workflow ensures mathematical correctness as the framework grows.

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

We have a complete, in-depth guide on how Gradience implements deep learning from scratch. Read through the guides below to understand how the framework works under the hood:

1. [Gradience Under the Hood: An Introduction](docs/01_Introduction.md)
2. [Tensors: The Smart Data Containers](docs/02_Tensors.md)
3. [The Computational Graph](docs/03_Computational_Graph.md)
4. [Functions and Context](docs/04_Functions_and_Context.md)
5. [The Autograd Engine](docs/05_Autograd_Engine.md)
6. [Neural Networks](docs/06_Neural_Networks.md)
7. [Code Execution & Broadcasting](docs/08_Under_The_Hood_Code_Execution.md)
8. [Testing & Gradcheck](docs/09_Testing_and_Gradcheck.md)
9. [Experiments and Validation](docs/10_Experiments_and_Validation.md)
10. [API Reference](docs/07_API_Reference.md)

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
