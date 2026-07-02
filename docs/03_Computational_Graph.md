# The Computational Graph

In the last section, we learned that Tensors remember where they came from by storing a link in their `_grad_fn` property. 

When you string a lot of math operations together, these links form a massive web. In computer science, we call this web a **Directed Acyclic Graph** (or DAG). In deep learning, we simply call it the **Computational Graph**.

## What is a Graph?

In programming, a graph is just a collection of "nodes" (points) connected by "edges" (lines). 

* **Directed** means the lines have arrows on them. They point in one specific direction.
* **Acyclic** means there are no circles or loops. If you follow the arrows, you will never end up back where you started.

## Building the Map

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

You have just built a Directed Acyclic Graph! The data flows forward, from the leaves (`x` and `y`) all the way to the final answer (`b`). 

## Why Do We Need This?

The whole point of AI is to find out how to change `x` and `y` to make `b` better. To do this, we need to find the derivative of `b` with respect to `x` and `y`.

If you remember your 12th grade calculus, you find the derivative of a complex equation by using the **Chain Rule**. You start at the outside of the equation and work your way inside, multiplying the derivatives together.

The Computational Graph is exactly what makes the Chain Rule possible in code! 

Because all the arrows in our graph point forward (from `x` to `b`), we can simply turn around and walk the arrows backward (from `b` to `x`). As we walk backward, we stop at every math node, calculate the derivative for that specific step, and multiply it with the derivative from the previous step.

## Dynamic vs Static Graphs

There is a very cool detail about how Gradience builds this graph. It builds it **dynamically**.

Dynamic means the graph is built on the fly, exactly as the code is executed. Every time you run your math loop, a brand new graph is constructed from scratch in the computer's memory. Once you go backward and calculate the derivatives, the graph is thrown in the trash to save memory. 

This is incredibly powerful because it means you can use normal Python `if` statements and `for` loops. If your code randomly decides to do addition instead of multiplication today, the framework just builds a different graph on the fly. 

Some systems use Static graphs, where you have to define the entire map before you can put any numbers into it. Gradience uses Dynamic graphs because they are so much easier for humans to read, write, and debug.

Now that we know how the map is built, we need to look closer at the math nodes themselves. We will do that in the next section: **[Functions and Context](04_Functions_and_Context.md)**.
