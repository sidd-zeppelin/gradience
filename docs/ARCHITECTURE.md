# Gradience Architecture

> **Building a modern deep learning framework from first principles.**

---

# Vision

Gradience is an educational yet practical deep learning framework implemented completely from scratch.

Its goal is **not** to replace PyTorch, TensorFlow or JAX.

Its goal is to answer one simple question:

> **"How does a modern deep learning framework actually work?"**

Every abstraction inside Gradience is intentionally small, modular and understandable.

Users should be able to read every major component and understand exactly why it exists and how it works.

---

# Why Gradience?

Modern frameworks provide incredible capabilities but often hide enormous implementation complexity.

A user can build a Transformer with only a few lines of code while never understanding:

- reverse-mode automatic differentiation
- computation graphs
- tensor broadcasting
- gradient accumulation
- parameter management
- optimization algorithms

Gradience exists to bridge that gap.

Rather than simply using deep learning, Gradience encourages understanding it.

---

# High-Level Architecture

```
                     User Code

                         │

                         ▼

              Neural Network API
──────────────────────────────────────────────

 Module
 Parameter
 Sequential
 Linear
 Activations
 Losses

                         │

                         ▼

              Tensor Operations
──────────────────────────────────────────────

 Add
 Subtract
 Multiply
 Divide
 Power
 Exp
 Log
 MatMul
 Mean
 Sum
 ...

                         │

                         ▼

             Automatic Differentiation
──────────────────────────────────────────────

 Tensor
 Function
 Context
 GraphNode
 AutogradEngine

                         │

                         ▼

                 NumPy Backend
```

For detailed information on specific subsystems, consult the **[Internals Documentation](internals/)**:
* **[Tensor Engine](internals/tensor.md)**
* **[Autograd & Computational Graphs](internals/autograd.md)**
* **[Module Abstractions](internals/module.md)**
* **[Broadcasting Math](internals/broadcasting.md)**
* **[Numerical testing & gradcheck](internals/testing.md)**

---

# Core Components

## Tensor

The central abstraction.

Responsibilities

- numerical data
- gradients
- computation graph references
- tensor API

Tensor intentionally does **not** perform differentiation.

---

## Function

Represents one differentiable mathematical operation.

Every operation implements

- forward()
- backward()

Examples

- AddOp
- MultiplyOp
- MatMulOp
- ReLUOp

---

## Context

Stores tensors required during the backward pass.

Only saves information necessary for gradient computation.

---

## GraphNode

Represents one node inside the computation graph.

Stores

- operation
- parent tensors
- saved context
- output tensor

---

## Autograd Engine

Responsible for

- graph traversal
- reverse topological ordering
- gradient propagation
- gradient accumulation

The engine is intentionally independent from neural network abstractions.

---

## Module

Base abstraction for every trainable component.

Examples

- Linear
- Sequential
- Activations
- Losses

---

## Parameter

Represents trainable tensors.

Allows Modules and Optimizers to automatically discover learnable parameters.

---

## Optimizer

Responsible only for updating parameters.

Current implementation

- SGD

Future implementations

- Adam
- AdamW
- RMSProp
- Momentum SGD

---

# Project Motto

Gradience exists to answer one question.

> **"What if every abstraction inside a modern deep learning framework could be understood from first principles?"**

Every line of code should move the project one step closer to answering that question.
