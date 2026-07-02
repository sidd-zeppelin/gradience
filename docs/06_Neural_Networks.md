# Neural Networks

We now have a fully functioning Automatic Differentiation framework. We have Tensors that track history, a dynamic Computational Graph, and an Autograd Engine that can perform the Chain Rule automatically. 

Now, we can build Artificial Intelligence.

## What is a Neural Network?

At its core, a Neural Network is just a massive, chained math equation. 

Imagine you want to predict the price of a house. You feed in the square footage (the input). The network multiplies that input by a number (a **weight**), adds another number (a **bias**), and spits out a price prediction.

At first, the weights and biases are just random numbers. The prediction will be completely wrong. But because we built our network using Gradience Tensors, every single math operation was recorded in the Computational Graph. 

We can compare our wrong prediction to the actual price to get a "Loss" value. Then, we simply call `loss.backward()`. 

Instantly, the Autograd Engine walks backward and calculates the derivative for every single weight and bias in the network. These derivatives tell us exactly how much to increase or decrease each weight to make the prediction more accurate next time.

## Abstracting the Math: Layers

While you could write out all this math manually (`y = x * weight + bias`), it gets very messy when you have millions of weights.

Gradience solves this by creating **Layers**. You can find these in the `gradience/nn/` folder. 

A Layer is just a Python class that holds the weight Tensors for you, and hides the math inside a method. In Gradience, the base class for this helps organize your neural network.

### The `Parameter` Class

If you look in `gradience/nn/parameter.py`, you will see a tiny but crucial class called `Parameter`. 
A `Parameter` is just a `Tensor` in disguise. When we build a Neural Network layer, we want the framework to know which Tensors are regular data (like an image), and which Tensors are the network's weights that need to learn. By wrapping our weights in the `Parameter` class, we flag them so the framework knows to treat them specially during the learning phase.

When you create a Linear Layer, it automatically generates the weight and bias Tensors with `requires_grad=True` turned on. When you pass data through the layer, it automatically does the multiplication and addition for you. 

## Activation Functions: The Power of Curves

If you only use multiplication and addition, your AI can only draw straight lines. If you try to predict something complicated (like recognizing a dog in a picture), a straight line is not enough. 

To solve this, we use **Activation Functions**. 

An Activation Function is just a math operation that introduces a curve (or non-linearity) into the equation. Famous examples include **ReLU** (Rectified Linear Unit), **Sigmoid**, and **Tanh**. 

Because these are just math operations, we implemented them in Gradience as standard `Function` classes (like `ReLUOp` in the `ops/` folder). They have a forward step (applying the curve) and a backward step (the derivative of the curve). 

By sandwiching Activation Functions between our Linear Layers, our Neural Network gains the ability to learn incredibly complex, curving patterns. 

## The Training Loop

When you put all of these concepts together, a typical AI training loop looks like this:

1. **Forward Pass**: Feed data through the Neural Network layers to get a prediction. The Computational Graph is built automatically.
2. **Calculate Loss**: Check how wrong the prediction is.
3. **Zero Gradients**: Clear out the old gradients from the last loop (remember Gradient Accumulation!).
4. **Backward Pass**: Call `.backward()` on the Loss. The Autograd Engine calculates the new derivatives.
5. **Optimizer Step**: A tool called an Optimizer looks at the new gradients and nudges the weights in the right direction.

And that is it! You now understand exactly how Gradience and deep learning work under the hood. You know how Tensors store data, how Graphs map equations, how Functions stash Context, how the Autograd Engine applies the Chain Rule, and how Neural Networks use it all to learn.

You have graduated from using AI as a black box to understanding it from first principles. Feel free to dive into the `gradience/` source code and explore the implementation for yourself!
