# The Autograd Engine

We have finally reached the brain of the entire framework. The **Autograd Engine** (Automatic Differentiation Engine). 

You can find the code for this in `gradience/autograd/autograd_engine.py`. 

When you finish running all your math and you finally call `.backward()` on your final Tensor, the Autograd Engine wakes up. Its job is to walk backward through the Computational Graph and trigger the Chain Rule. 

Let us break down exactly how it does this, step by step.

## Step 1: The Topological Sort

Imagine you are standing at the end of a maze (the final answer Tensor) and you need to walk backward to the start (the leaf Tensors). But there is a strict rule: you are not allowed to visit a room until you have visited all the rooms that lead into it. 

To solve this, the Autograd Engine uses a famous computer science algorithm called a **Topological Sort**.

It explores the entire graph and creates a perfectly ordered, flat list of all the Functions. In this list, a parent node will always appear *after* the child nodes that depend on it. This guarantees that as we walk down the list, we do the calculus in the exact right order.

## Step 2: Seeding the Gradient

To start the Chain Rule, you need a starting number. 
The derivative of a variable with respect to itself is always `1.0`. 
So, the engine takes the final answer Tensor and artificially sets its gradient to `1.0`. It then passes this `1.0` into the very first Function on our sorted list.

## Step 3: The Backward Loop

Now, the engine simply loops through the sorted list of Functions. For every Function, it does the following:

1. It takes the incoming gradient (we call this `grad_output`).
2. It calls the `backward(ctx, grad_output)` method on the Function.
3. As we learned in the previous section, the Function uses the Chain Rule and its Context backpack to calculate the local derivatives for its inputs.
4. The Function spits these new derivatives back out.

The engine takes these new derivatives and passes them down to the next Functions in the list. This is the Chain Rule happening automatically in a loop!

## Step 4: Gradient Accumulation

There is one major gotcha in calculus that the engine has to handle. 

What if you use the same variable twice in an equation? Like `z = x * x`.
In the graph, `x` is used in two different places. When the engine walks backward, it will calculate a derivative for the left `x` and a completely different derivative for the right `x`. 

If the engine just overwrites the `_grad` property on the `x` Tensor, one of the derivatives will be lost! 

To fix this, the engine uses **Gradient Accumulation**. When it calculates a derivative for a Tensor, it does not overwrite the old one. It *adds* them together. `tensor.grad = tensor.grad + new_grad`. 

This is why, in Gradience, you always have to call `zero_grad()` before doing a new math loop, otherwise your new gradients will be added on top of your old ones!

## The Result

By the time the `for` loop reaches the end of the sorted list, every single leaf Tensor will have its final, perfect derivative sitting inside its `grad` property. 

The Autograd Engine has successfully solved massive, complex calculus equations in a fraction of a second.

With this math engine complete, we can finally build Artificial Intelligence. We will look at that in the final section: **[Neural Networks](06_Neural_Networks.md)**.
