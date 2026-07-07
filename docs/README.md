# Gradience Under the Hood: An Introduction

Welcome! If you want to understand how neural networks and deep learning actually work behind the scenes, you are in the right place. We are going to break down the complex math and computer science into simple level ideas.

## Why Gradience?

Gradience is a deep learning framework built entirely from scratch. Its purpose is educational: to help you understand exactly what happens inside a neural network without being overwhelmed by millions of lines of complex code.

## The Perfect Learning Tool

Because Gradience is tiny and readable, you can open any file and immediately see the math in action. It strips away the unnecessary complexity and focuses entirely on the core concepts that make Artificial Intelligence possible. By reading the code in Gradience, you will learn exactly how deep learning works from first principles.

## The Core Concept: Automatic Differentiation

The secret to all modern Artificial Intelligence is **Calculus**. 

Specifically, AI learns by looking at how wrong its predictions are, and then using a derivative to figure out which direction to adjust its numbers. 

If you have a function `y = 3 * x`, the derivative tells you that if you increase `x` by 1, `y` will increase by 3. In AI, we have functions with millions of `x` variables. We need a way to calculate the derivative for every single one of them automatically.

This process is called **Automatic Differentiation** (or Autograd for short). Gradience uses a specific type called "Reverse Mode Automatic Differentiation". 

## How This Documentation is Structured

We have reorganized our guides into high-level specs, internal deep-dives, and hands-on tutorials:

### High-Level Documentation
* **[Architecture Guide](ARCHITECTURE.md)**: Conceptual diagram and overview of the pipeline layers.
* **[Design Decisions](DESIGN.md)**: Design trade-offs (e.g. why we choose NumPy, custom functions vs. fused operations).
* **[Contributing Guidelines](CONTRIBUTING.md)**: Quick-start specs for implementing new mathematical operations and NN layers.
* **[Changelog](CHANGELOG.md)**: Detailed log of all features, improvements, and bug fixes.

### Subsystem Internals (Deep Dives)
* **[Tensor Engine](internals/tensor.md)**: Anatomy of our wrapper, fields (`_data`, `requires_grad`, etc.), and dunder overloads.
* **[Autograd & Computational Graphs](internals/autograd.md)**: Dynamic tape construction, reverse topological sorting, and gradient accumulation.
* **[Module Abstractions](internals/module.md)**: Reflective parameter registration, Sequential layers, and train/eval toggles.
* **[Broadcasting Math](internals/broadcasting.md)**: Detailed trace of forward/backward operations and gradient shape reconciliation.
* **[Numerical testing & gradcheck](internals/testing.md)**: How numerical finite difference testing is used to verify analytical gradients.

### Step-by-Step Tutorials
* **[Tutorial 1: Training Your First Model](tutorials/tutorial_1_linear_regression.md)**: Step-by-step implementation of a linear regression training loop.
* **[Tutorial 2: Building Deep Neural Networks](tutorials/tutorial_2_deep_learning.md)**: Stack activation layers and use the Adam optimizer to solve the XOR classification boundary.

### API Cheatsheet
* **[API Reference](API_Reference.md)**: Full signatures and parameters of all operators, layers, and optimizers.

Take your time. Read through these pages, and open the code files in the `gradience/` folder as you go. You will see that AI is just a bunch of simple math concepts chained together!
