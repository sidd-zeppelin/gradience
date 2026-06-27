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

Rather than treating machine learning as a black box, this project explores how frameworks like PyTorch work internally by implementing every major component ourselves.

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

### Testing Infrastructure

* Unit tests
* Autograd tests
* Numerical gradient checking
* Multi-input gradient verification

---

## Project Architecture

```text
gradience/
│
├── autograd/
│   ├── engine.py
│   ├── context.py
│   ├── function.py
│   └── graph_node.py
│
├── ops/
│   ├── add.py
│   └── multiply.py
│
├── testing/
│   └── gradcheck.py
│
├── tensor.py
│
└── ...
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

* [ ] Subtraction
* [ ] Division
* [ ] Power
* [ ] Exponential
* [ ] Logarithm
* [ ] Broadcasting
* [ ] Matrix multiplication

### Phase 3 — Neural Network API

* [ ] Parameter
* [ ] Module
* [ ] Linear layer
* [ ] Sequential container
* [ ] Activation functions
* [ ] Loss functions

### Phase 4 — Training

* [ ] Optimizers
* [ ] SGD
* [ ] Adam
* [ ] Learning rate schedulers
* [ ] End-to-end neural network training

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

pytest will be configured soon.

---

## Documentation

The Docs will be worked upon soon.

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

Whether you're interested in machine learning, software engineering, numerical computing, or simply learning how deep learning frameworks work internally, feel free to open an issue or submit a pull request.

---

## License

This project is released under the MIT License.
