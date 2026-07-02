# Functions and Context

In the previous section, we talked about the nodes in the Computational Graph. When you multiply two Tensors together, a "Multiplication Node" is created in the background.

In Gradience, these nodes are called **Functions**. You can find the base blueprint for them in `gradience/autograd/function.py`, and the specific math operations (like addition, multiplication, sine, cosine) in the `gradience/ops/` folder.

## The Two Jobs of a Function

Every Function in our framework must know how to do two completely different things:

1. **Forward**: Do the normal math. If it is a Multiplication Function, it multiplies the numbers. 
2. **Backward**: Do the calculus. It calculates the derivative of that specific math operation using the rules you learned in high school.

## The Problem with Calculus

There is a tricky problem when writing the Backward step. 

Let us say your function is `y = x^2` (x squared). The derivative of `x^2` is `2 * x`. 
If you are doing the Backward step, you need to know what `x` is so you can multiply it by 2. But the Forward step already finished a long time ago! How does the Backward step know what `x` was?

## The Solution: The Context Backpack

To solve this, Gradience introduces an object called the **Context**. You can find our version in `gradience/autograd/context.py`.

Think of the Context as a small backpack that the Function wears. 

During the **Forward** step, the Function does its normal math, but right before it finishes, it takes any numbers it might need for calculus later (like `x`) and shoves them into the Context backpack. In code, this looks like `ctx.save_for_backward(x)`.

Much later, when the framework is running the **Backward** step, the Function reaches into its backpack, pulls out the saved variables (`saved_tensors = ctx.saved_tensors`), and uses them to calculate the exact derivative. 

## Putting it Together

Here is a simplified look at how the Multiplication Function (`x * y`) is written in code:

```python
class MultiplyOp(Function):
    
    @staticmethod
    def forward(ctx, x, y):
        # 1. Do the normal math
        result = x.data * y.data
        
        # 2. Save x and y in the backpack for later!
        ctx.save_for_backward(x.data, y.data)
        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # 1. Take x and y out of the backpack
        x_data, y_data = ctx.saved_tensors
        
        # 2. Do the calculus! 
        # The derivative of (x * y) with respect to x is y.
        # The derivative of (x * y) with respect to y is x.
        
        grad_x = grad_output * y_data
        grad_y = grad_output * x_data
        
        return grad_x, grad_y
```

This brilliant design is what makes Gradience so modular. You can add any crazy math operation you want to the framework, as long as you provide a `forward` method and a `backward` method using a Context backpack.

Now that we have Tensors, a Graph, and Functions that know calculus, we just need a machine to put it all together. We will explore that in the next section: **[The Autograd Engine](05_Autograd_Engine.md)**.
