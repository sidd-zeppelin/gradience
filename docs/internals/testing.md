# Testing: How We Prove the Math Works

When you build a deep learning framework from scratch, a small mistake in a single calculus derivative can break everything. A neural network might run without crashing, but it will silently fail to learn anything.

So how do we prove that Gradience is mathematically perfect?

We use a technique called **Numerical Gradient Checking**. You can find the implementation in `gradience/testing/gradcheck.py`.

## The Two Ways to Find a Derivative

There are two ways to find the derivative of a function.

### 1. Analytical (The Autograd Way)
This is what Gradience does. It uses the exact, perfect rules of calculus (like the Chain Rule) to calculate the derivative step-by-step. We call this the "Analytical" gradient.

### 2. Numerical (The Computer Way)
The second way is to cheat. 
A derivative is just a measure of how much a tiny change in the input affects the output. We can ask the computer to calculate the function twice: once where we add a tiny, microscopic number to the input (`eps = 0.00001`), and once where we subtract it. 

We can then see how much the output changed. This is called the "Numerical" gradient. It is not perfectly exact, but it is extremely close.

## The `gradcheck` Function

Our `gradcheck` function uses the Numerical method to test the Analytical method.

Here is exactly what it does when you ask it to test a math operation:

1. **The Analytical Pass**: It takes your input Tensors, runs them through the math operation, and calls `.backward()`. This asks the Autograd Engine to calculate the exact derivatives.
2. **The Numerical Pass**: It takes the same math operation, and nudges the inputs up and down by `0.00001` (using a formula called the "centered difference"). It calculates the estimated derivative.
3. **The Comparison**: It takes the exact gradient and subtracts the estimated gradient. 
4. **The Verdict**: If the difference between them is smaller than a tiny tolerance (`1e-6`), the test passes! If the difference is larger, it means our calculus in the `backward` method is mathematically wrong, and the test crashes with an error.

## Why This Matters

Every single math operation in the `gradience/ops/` folder has an automated test that runs `gradcheck`. 

Before any code is accepted into Gradience, it must pass these tests. This guarantees that our Autograd Engine is completely bulletproof, and allows us to trust the framework when we build massive Neural Networks!
