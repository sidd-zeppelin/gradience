# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
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

### Fixed
- Fixed an unreachable and duplicated backward pass definition in `AutogradEngine`.
- Fixed a gradient tensor iteration bug during the backward pass of unary operations.