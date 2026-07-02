# Autograd Engine & The Computational Graph

This document details how the computational graph is built and how the autograd engine executes the backward pass to calculate gradients.

---

## Part 1: The Computational Graph

When you string a lot of math operations together, their links form a web. In computer science, we call this web a **Directed Acyclic Graph** (or DAG). In deep learning, we simply call it the **Computational Graph**.

### What is a Graph?
In programming, a graph is just a collection of "nodes" (points) connected by "edges" (lines). 
* **Directed** means the lines have arrows on them. They point in one specific direction.
* **Acyclic** means there are no circles or loops. If you follow the arrows, you will never end up back where you started.

### Building the Map
Let us look at a simple math equation:
```python
x = Tensor(2.0)
y = Tensor(3.0)
a = x * y
b = a + 5
```

As Python runs this code line by line, it is secretly building a graph in the background.
1. It creates `x` and `y`. These are leaf Tensors.
2. It hits the multiplication step `x * y`. The framework creates a new math object (a Multiplication Node) and points arrows from `x` and `y` into this node. The result that pops out is `a`.
3. It hits the addition step `a + 5`. The framework creates an Addition Node. It points an arrow from `a` to this new node. The result that pops out is `b`.

The data flows forward, from the leaves (`x` and `y`) all the way to the final answer (`b`). 

### Why Do We Need This?
To update neural network weights, we need to find the derivative of the loss with respect to all parameters. This is done by applying the **Chain Rule** from calculus. We start at the outside of the equation and work our way inside, multiplying the derivatives together.
The Computational Graph makes the Chain Rule possible in code! Because all the arrows in our graph point forward, we can simply walk the arrows backward, calculate the derivative for each step, and multiply them.

### Dynamic vs Static Graphs
Gradience builds the graph **dynamically** (define-by-run). Every time you run a forward pass, a brand new graph is constructed from scratch in memory. Once you run `.backward()`, the graph is cleared to save memory. This allows using normal Python `if` statements and `for` loops in model forward passes.

---

## Part 2: Functions and Context

In Gradience, the math nodes in the graph are represented by subclasses of `Function` (found in `gradience/ops/`).

### The Two Jobs of a Function
Every Function in our framework must define:
1. **Forward**: Do the normal math (e.g. multiply inputs using NumPy).
2. **Backward**: Do the calculus (calculate local derivatives and multiply by the incoming gradient).

### The Context Backpack
During the backward step, the derivative calculation might depend on the forward inputs (e.g. the derivative of $x^2$ is $2x$). To remember what the inputs were, every operation has a `Context` backpack (`gradience/autograd/context.py`).
During `forward`, the function calls `ctx.save_for_backward(*tensors)`. During `backward`, it retrieves them via `ctx.saved_tensors`.

Here is the Multiplication operation as an example:
```python
class MultiplyOp(Function):
    @staticmethod
    def forward(ctx, x, y):
        result = x * y
        ctx.save_for_backward(x, y)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output * y, grad_output * x
```

---

## Part 3: The Autograd Engine

The `AutogradEngine` (`gradience/autograd/autograd_engine.py`) walks backward through the graph to trigger the Chain Rule.

### Step 1: Topological Sort
To run backpropagation, we must process nodes in reverse topological order. This guarantees that a node's gradient is only computed *after* we have processed all nodes depending on it. The engine does this using DFS to flatten the graph.

### Step 2: Seeding the Gradient
The derivative of a variable with respect to itself is `1.0`. We seed the terminal node's gradient to `1.0` (matching the terminal tensor's shape) to kick off the chain.

### Step 3: The Backward Loop
The engine loops backward through the topological list. For each node, it:
1. Takes the accumulated gradient (`grad_output`).
2. Calls the operation's `backward(ctx, grad_output)`.
3. Propagates the returned local gradients to the parents.

### Step 4: Gradient Accumulation
If a variable is used multiple times (e.g., `z = x * x`), the engine receives multiple gradients for it. To avoid overwriting them, the engine **accumulates** (adds) them: `tensor.grad = tensor.grad + new_grad`.
