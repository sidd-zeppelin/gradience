# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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