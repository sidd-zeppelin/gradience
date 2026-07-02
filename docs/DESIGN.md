# Design Philosophy & Decisions

This document details the architectural decisions and trade-offs driving the development of the Gradience framework.

---

## 1. Educational Clarity vs. Raw Speed

The primary target of Gradience is to teach how automatic differentiation and neural network layers function under the hood.

* **Trade-off:** We rely on NumPy for low-level matrix computations rather than custom C++/CUDA code. 
* **Reasoning:** Pure Python/NumPy logic is extremely readable. Writing CUDA kernels would obscure the underlying algorithmic implementations for the learner. 

## 2. Separation of Tensors and Operations

The `Tensor` class is kept as a simple, stateless wrapper around a data array. All mathematical operations are defined in separate classes inheriting from `Function` (e.g., `AddOp`, `MultOp`, `CrossEntropy`).

* **Reasoning:** This prevents the `Tensor` class from becoming a monolithic file containing thousands of lines of unrelated mathematical code. It makes adding a new operation as simple as dropping a new file into `gradience/ops/`.

## 3. Fused Operations for Numerical Stability

While most layers (such as `BatchNorm1d`) are built by composing primitive tensor operations, some components must be "fused" for numerical stability.

* **Example:** `CrossEntropyLoss` and `BCEWithLogitsLoss`.
* **Reasoning:** If implemented as chains of individual operations (e.g., Sigmoid followed by Log followed by NLL), intermediate probabilities could overflow or saturate to exactly `0.0` or `1.0`, leading to `NaN` gradients. Fused operations leverage numerical stablization techniques (like max-shifting in Log-Sum-Exp) inside a single forward/backward pass.

## 4. In-Place Parameter Mutation

During parameter updates, optimizers mutate the private `._data` array directly (e.g., `param._data -= lr * param.grad.data`).

* **Reasoning:** Mutating the underlying NumPy data avoids reallocating `Tensor` objects or breaking object references held by `Module` classes, ensuring parameter changes propagate seamlessly across layers.

## 5. Topological Pruning

The `AutogradEngine` prunes subgraphs that do not require gradients.

* **Reasoning:** Intermediate tensors created from operations where all inputs have `requires_grad=False` will not have a `GraphNode` generated. This prevents the topological sorter from visiting dead branches, speeding up forward passes and saving immense memory.
