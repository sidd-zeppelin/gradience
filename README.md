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

To prove that the Gradience autograd engine is mathematically identical to industry standards, we benchmarked models on standard datasets.

Both frameworks were instantiated identically, using deterministic seeds, identical batched datasets, identical hyperparameters, and identical initial random weights transferred memory-for-memory.

### 1. MLP Benchmark (MNIST)

We benchmarked a 2-Layer Multi-Layer Perceptron (784 to 128 to 10) on the MNIST dataset using SGD with a learning rate of 1.0.

| Metric | Gradience (NumPy Backend) | PyTorch (C++ ATen Backend) |
| :--- | :--- | :--- |
| **Forward Pass Time** | ~0.336 s | ~0.147 s |
| **Backward Pass Time** | ~0.868 s | ~0.193 s |
| **Total Training Time** | ~1.567 s | ~0.750 s |
| **Final Test Accuracy** | **90.00%** | **90.00%** |

### 2. CNN Benchmark (MNIST & CIFAR-10)

We benchmarked LeNet-5 and AlexNet to verify the correctness of our convolutional layers, pooling operations, and tensor routing.

* **LeNet-5 on CIFAR-10**: The forward activations, loss, and backpropagation parameter gradients match PyTorch to within a tolerance of 1e-12.
* **AlexNet on CIFAR-10**: The complex composition of convolutions, pooling, reshaping, and dropout layers produces identical logits and parameter gradients compared with PyTorch.

| Operation / Metric | LeNet-5 Maximum Difference | AlexNet Maximum Difference |
| :--- | :--- | :--- |
| **Forward Logit Difference** | < 1.00e-12 | < 1.00e-12 |
| **Loss Value Difference** | < 1.00e-12 | < 1.00e-12 |
| **Parameter Gradient Difference** | < 1.00e-12 | < 1.00e-12 |

**Conclusion:**
Gradience achieves exactly 100% mathematical identicality to PyTorch across both fully connected and convolutional neural network architectures. As an educational pure-Python framework, retaining competitive CPU speed is an outstanding validation of NumPy's underlying optimizations combined with Gradience's efficient topological graph traversals.

---

## Features

### Core Engine
* **NumPy-Backed Tensors**: Fast, underlying C-optimized matrix math.
* **Dynamic Computation Graph**: Builds reverse-mode automatic differentiation on the fly.
* **Broadcasting**: Natively supports full dimensional broadcasting during forward and backward passes.
* **Gradient Accumulation**: Handles arbitrary node branching and re-convergence.

### Mathematical Operations
* Arithmetic: `add`, `sub`, `mul`, `div`, `pow`, `neg`
* Reshaping & Reductions: `sum`, `mean` (with `keepdims` support), `reshape`, `flatten`
* Transcendental: `exp`, `log`, `sqrt`
* Trigonometry: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
* Matrix Math: `matmul` (`@`)

### Neural Network API (`gradience.nn`)
* **Modules & Parameters**: Abstract base classes for building stateful models.
* **Layers**: `Linear` (Dense), `Conv2D`, `MaxPool2D`, `AdaptiveAvgPool2D`, `Dropout`, `BatchNorm1d`, `LayerNorm`.
* **Containers**: `Sequential` models.
* **Activations**: `ReLU`, `Sigmoid`, `Tanh`.
* **Loss Functions**: `MSELoss`, `L1Loss`, `CrossEntropyLoss` (Fused), `BCEWithLogitsLoss` (Fused).
* **Initializers**: `he_uniform`, `xavier_uniform`, `normal`, `zeros`, etc.
* **Model Architectures**: `AlexNet`.

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

Gradience features a comprehensive deep learning textbook curriculum for beginners. All mathematics and code implementations are derived from first principles.

### Deep Learning Textbook Chapters
* **[Syllabus & Roadmap Index](docs/README.md)**: Conceptual index and syllabus of the learning path.
* **[Chapter 1: Introduction to Deep Learning](docs/chapters/chapter_1_introduction.md)**: Biological/artificial neurons, perceptrons, multi-layer perceptrons, and universal approximation.
* **[Chapter 2: Building the Foundations](docs/chapters/chapter_2_foundations.md)**: Tensors, computational graphs, backpropagation, and reverse-mode automatic differentiation (autograd).
* **[Chapter 3: Neural Networks](docs/chapters/chapter_3_neural_networks.md)**: Linear layers, activations (ReLU, Sigmoid, Tanh), loss functions (MSE, BCE, Cross-Entropy), and MLPs.
* **[Chapter 4: Optimization](docs/chapters/chapter_4_optimization.md)**: Gradient descent variants, SGD (momentum, Nesterov, weight decay), Adagrad, RMSprop, Adam, AdamW, and learning rate scheduling.
* **[Chapter 5: Training Neural Networks](docs/chapters/chapter_5_training.md)**: Weight initialization (Xavier, Kaiming/He), normalization layers (BatchNorm, LayerNorm), regularization (Dropout), datasets, and dataloaders.
* **[Chapter 6: Computer Vision Foundations](docs/chapters/chapter_6_computer_vision.md)**: Image representations, 2D convolutions, padding, stride, and pooling (MaxPool2D, AdaptiveAvgPool2D).
* **[Chapter 7: CNN Architectures](docs/chapters/chapter_7_cnn_architectures.md)**: Milestone vision networks: LeNet-5 for MNIST and original split-device dual-stream AlexNet (channel concatenation via ConcatOp).

### Example Notebooks
* **[Graph Visualization Demo](examples/visualize_graph.ipynb)**: Walkthrough of computation graph extraction, rendering, and file export for simple and complex mathematical functions.
* **[Linear Regression Demo](examples/linear_regression.ipynb)**: Training a simple linear regression model to fit synthetic 1D data.
* **[XOR Classification Demo](examples/train_xor.ipynb)**: Solving the classic non-linear XOR classification problem with a multi-layer perceptron (MLP).
* **[MNIST Classification Demo](examples/train_mnist.ipynb)**: Training a two-layer MLP on the MNIST handwritten digit database, showing step-for-step correctness compared with PyTorch.
* **[AlexNet Demo](examples/train_alexnet.ipynb)**: Notebook training the standard single-stream AlexNet model on a CIFAR-10 dataset subset, comparing mathematical correctness and performance directly against PyTorch.
* **[AlexNet Dual-Stream Demo](examples/train_alexnet_parallel.ipynb)**: Notebook training the historical dual-stream (split-device parallel) AlexNet model on a CIFAR-10 dataset subset, comparing logit outputs, gradients, and loss convergence directly against PyTorch (running on CUDA/GPU).
* **[LeNet-5 Demo](examples/train_lenet.ipynb)**: Notebook training the classic LeNet-5 architecture on the CIFAR-10 dataset, comparing output logits, gradients, and convergence directly against PyTorch.

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
