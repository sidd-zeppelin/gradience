# Contributing to Gradience

Contributions to Gradience are welcome! To maintain software quality and mathematical correctness, please adhere to the following guidelines.

---

## 1. Code Style & Architecture
* Follow PEP 8 guidelines.
* Preserve separation of concerns. Keep operations in `gradience/ops/`, layers in `gradience/nn/layers/`, and optimizers in `gradience/optim/`.
* Write clean, documented code. Every new mathematical module must have a clear docstring explaining its mathematical formulation.

## 2. Implementing a New Operation
To add a new mathematical operation to the framework:
1. Create a subclass of `Function` in `gradience/ops/`.
2. Implement `@staticmethod` methods for `forward(ctx, ...)` and `backward(ctx, grad_output)`.
3. Wrap inputs using raw NumPy `.data` fields in `forward`, and save necessary variables for backward pass using `ctx.save_for_backward(...)`.
4. Register the new operation inside the `Tensor` class in `gradience/tensor.py` by overloading the corresponding method or operator.
5. Export the new operation in `gradience/ops/__init__.py`.

## 3. Implementing a New Layer
To add a neural network layer:
1. Subclass `Module` from `gradience.nn.module`.
2. Define trainable weights as `Parameter` instances inside the `__init__` constructor.
3. Implement the `forward(self, x)` method using the framework's mathematical operators.
4. Ensure the module handles both `self.training = True` and `self.training = False` states appropriately if its behavior differs during evaluation (e.g., Dropout, BatchNorm).

## 4. Testing Requirements
We enforce **100% test coverage** for all new code.
* Place unit tests inside the `tests/` directory.
* Every operation must be validated using numerical gradient checking (`gradcheck`). See existing tests in `tests/test_ops.py` for how to use `gradcheck`.
* Run the tests using:
  ```bash
  uv run python -m pytest tests/
  ```
