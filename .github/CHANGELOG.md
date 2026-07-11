# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-07-11
### Added
- Created **7-Chapter First-Principles Deep Learning Curriculum** as a comprehensive textbook under `docs/chapters/` following a 13-section roadmap layout.
- Implemented **Original Split-GPU AlexNet Architecture** (`OriginalAlexNet`) to model Krizhevsky's historical 2012 dual-stream routing design from first principles.
- Created `ConcatOp` autograd function in `gradience/ops/concat.py` supporting backpropagation of concatenated feature maps.
- Exposed a public `concat()` tensor helper in the main namespace.
- Added a comprehensive Jupyter Notebook demo (`examples/train_alexnet_parallel.ipynb`) comparing training convergence, logs, gradients, and loss plots of `OriginalAlexNet` against PyTorch GPU/CUDA on CIFAR-10.
- Implemented **2D Pooling Support** (`MaxPool2D`, `AdaptiveAvgPool2D`) with corresponding autograd backward operations.
- Added **Reshaping & Flattening Operations** (`reshape`, `flatten`) to allow flexible tensor restructuring.
- Added standard single-stream **AlexNet Architecture** model to verify complex composition of convolution, pooling, and linear layers.
- Implemented **2D Convolution Support** (`Conv2D`) using a naïve first-principles loop-based algorithm optimizing for correctness and educational value.
- Added verification test suite for pooling, reshaping, convolution, and AlexNet models.
- Refactored parameter `_calculate_fan` in `gradience/nn/init.py` to support multidimensional parameter tensors (4D convolution kernels) during initialization.
- Integrated and exposed `Conv2D` under the main `gradience.nn` package interface.

## [1.1.0] - 2026-07-06
### Added
- Implemented decoupled **Computational Graph Visualization** subsystem (`gradience.visualization`) using a bipartite generic graph representation (`GraphNode`, `GraphEdge`, `ComputationGraph`) and a post-order style DFS autograd graph extractor (`GraphExtractor`).
- Created a `GraphRenderer` that constructs visual diagrams using Graphviz `Digraph` rendering, modeling operations as boxes and tensors as ellipses with shape/type metadata and scalar previews.
- Integrated `Tensor.visualize()` as a main public method on the `Tensor` class.
- Authored a comprehensive visualization guide (`docs/visualization.md`) outlining architecture design decisions, algorithms, and future extensibility.
- Added comprehensive unit and integration tests in `tests/test_visualization.py`.
- Created an interactive Jupyter Notebook example (`examples/visualize_graph.ipynb`) demonstrating simple and complex mathematical function visualizations (including division, exponentials, roots, branching, and shared parameters).

### Changed
- Exposed submodules `nn`, `optim`, `testing`, and `visualization` directly under the main `gradience` namespace in `gradience/__init__.py`.
- Cleaned up subpackage public APIs by populating empty `__init__.py` files in subdirectories (e.g. `nn/activations/`, `nn/layers/`, `nn/losses/`, `nn/containers/`, `testing/`) with clean imports and explicit `__all__` lists.

## [1.0.0] - 2026-07-03
### Added
- Implemented Neural Network Layers: `Dropout`, `BatchNorm1d`, and `LayerNorm`.
- Implemented advanced optimization algorithms: `Adam`, `AdamW`, `RMSprop`, and `Adagrad`.
- Enhanced `SGD` with support for `momentum`, `nesterov` acceleration, and `weight_decay`.
- Implemented robust loss functions: `CrossEntropyLoss` and `BCEWithLogitsLoss` using numerically stable fused `ops`, alongside standard `L1Loss`.
- Added the `Abs` primitive operation and `Tensor.abs()` method.
- Added comprehensive step-by-step tutorials (`tutorial_1_linear_regression.md`, `tutorial_2_deep_learning.md`) bridging the gap between theoretical knowledge and practical framework usage.
- Restructured framework documentation into a clean architecture containing design philosophy (`DESIGN.md`), roadmap (`ROADMAP.md`), contribution guidelines (`CONTRIBUTING.md`), high-level architecture (`ARCHITECTURE.md`), and subsystem internals (`docs/internals/`).

### Fixed
- Fixed a critical topological sort bug in the `AutogradEngine` where `GraphNode` objects were unconditionally instantiated for intermediate tensors even when `requires_grad=False`.

## [0.1.0] - 2026-07-03
### Added
- Initial core tensor operations and autograd engine.
- Basic test suite for the framework.
- Finished the Autograd Engine completely.
- implemented all basic elementary operations for the tensor objects.
- Implemented Exponential (`exp`), Logarithm (`log`), and Square Root (`sqrt`) operations.
- Implemented core trigonometric operations (`sin`, `cos`, `tan`) and their inverses (`asin`, `acos`, `atan`).
- Implemented Reductions (`sum`, `mean`) with robust gradient broadcasting across dimensions.
- Implemented Matrix Multiplication (`matmul`) and overloaded python's `@` operator.
- Implemented core Activation Functions (`ReLU`, `Sigmoid`, `Tanh`) for Neural Networks.
- Enhanced NumPy interoperability by configuring `__array_priority__` and `__array_ufunc__`.
- Refactored test suite to use a uniform `conftest.py` with custom assertion fixtures.
- Achieved strictly verified 100% test coverage across the entire codebase.
- Implemented tensor broadcasting and backward pass unbroadcasting for element-wise operations.
- Implemented Parameter abstraction for tracking learnable weights in Neural Networks.
- Implemented the base `Module` class and the `Linear` (Dense) layer.
- Implemented neural network weight initialization functions (`he_uniform`, `xavier_uniform`, etc).
- Implemented the `Sequential` container module and verified parameter discovery and internal routing.
### Changed
- Reorganized the repository architecture to introduce dedicated subpackages for `nn` components (`activations/`, `containers/`, `layers/`, `losses/`) and optimizers (`optim/`).
- Converted end-to-end integration examples (`examples/linear_regression.ipynb`, `examples/train_xor.ipynb`, and `examples/train_mnist.ipynb`) into comprehensive Jupyter Notebooks featuring training graphs, data visualizations, and decision boundary plots.
- Added 1:1 PyTorch re-implementations inside all example notebooks to mathematically validate Gradience's correctness.
- Authored a comprehensive `10_Experiments_and_Validation.md` wiki page to document the framework's mathematical proofs and experimental outcomes.
- Implemented Neural Network `Activation` module layers (`ReLU`, `Sigmoid`, `Tanh`).
- Implemented Neural Network `Loss` module layers (`MSELoss`).
- Implemented the `Optimizer` base class and the `SGD` (Stochastic Gradient Descent) optimization algorithm.
- Moved `Sequential` to `gradience/nn/containers/sequential.py`.
- Moved `Linear` to `gradience/nn/layers/linear.py`.
- Refactored `Sequential` parameter storage to use a private `_modules` attribute.
- Enhanced `Sequential` parameter discovery algorithm to support nested lists and tuples natively.
- Added python container dunder methods (`__len__`, `__getitem__`, `__repr__`) to `Sequential` to allow indexing and clean printing.
### Fixed
- Fixed an unreachable and duplicated backward pass definition in `AutogradEngine`.
- Fixed a gradient tensor iteration bug during the backward pass of unary operations.
- Fixed a mathematical error in the subtraction operation backward pass and its associated unit test.
- Fixed an Autograd Engine bug to prevent wasteful gradient accumulation for tensors with `requires_grad=False`.
- Fixed a bug in `Module.parameters()` where the module iterated over `.items()` instead of `.values()`.
- Fixed a tuple unpacking bug in the `he_uniform` initializer that crashed `Linear` layer creation.