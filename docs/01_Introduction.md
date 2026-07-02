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

## How This Guide is Structured

We will explore every single concept that makes this possible. We have broken them down into separate pages so you can learn them one by one.

1. **[Tensors](02_Tensors.md)**: The data containers that hold our numbers and track their own history.
2. **[The Computational Graph](03_Computational_Graph.md)**: How the computer builds a map of your math equations.
3. **[Functions and Context](04_Functions_and_Context.md)**: How we teach the computer to do basic math and basic calculus at the same time.
4. **[The Autograd Engine](05_Autograd_Engine.md)**: The machine that walks backward through the map to calculate the derivatives.
5. **[Neural Networks](06_Neural_Networks.md)**: How we put it all together to build AI that learns.
6. **[Code Execution & Broadcasting](08_Under_The_Hood_Code_Execution.md)**: An end-to-end trace of a math operation and how we solve NumPy broadcasting.
7. **[Testing & Gradcheck](09_Testing_and_Gradcheck.md)**: How we mathematically prove that our framework is bug-free.
8. **[API Reference](07_API_Reference.md)**: Your cheat sheet for all the tools and math operations in Gradience.

Take your time. Read through these pages, and open the code files in the `gradience/` folder as you go. You will see that AI is just a bunch of simple math concepts chained together!
