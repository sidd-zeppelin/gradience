# Under the Hood: End-to-End Code Execution

Up until now, we have talked about the concepts of deep learning. But how does Gradience *actually run this code*?

In this section, we will walk through the exact lifecycle of a single math equation. We will trace `z = x * y` from the moment you type it, all the way to the moment `z.backward()` finishes calculating the gradients. 

By the end of this page, you will understand exactly how the `Context` bag works, how `apply()` connects the graph, and how broadcasting is solved.

---

## Part 1: The Forward Pass (`z = x * y`)

Imagine you type the following code:
```python
x = Tensor([2.0], requires_grad=True)
y = Tensor([3.0], requires_grad=True)
z = x * y
```

### 1. Intercepting the Math
When Python sees the `*` symbol, it looks at the `x` Tensor and triggers a special method called `__mul__`. 

If you look in `gradience/tensor.py`, you will see:
```python
def __mul__(self, other):
    from gradience.ops.multiply import MultOp
    return MultOp.apply(self, other)
```
Instead of multiplying the numbers immediately, the Tensor hands the job over to the `MultOp` function by calling its `apply()` method.

### 2. The `apply()` Pipeline
The `apply()` method (found in `gradience/autograd/function.py`) is the most important piece of glue in the entire framework. It is responsible for running the math and building the map. 

Here is exactly what `apply()` does, in order:

*   **Unpacking**: It opens the Tensors and takes out the raw NumPy arrays (`x.data` and `y.data`).
*   **The Context Bag**: It creates a brand new, empty `Context` object. Think of this as a small backpack that this specific multiplication operation will carry.
*   **Running Forward**: It calls the `forward(ctx, x_data, y_data)` method inside `MultOp`.

### 3. Inside the `forward` Method
Now we are inside the `forward` method of `MultOp`. 
```python
@staticmethod
def forward(ctx, x, y):
    # 1. Do the normal math using NumPy
    result = x * y
    
    # 2. Put variables in the Context bag!
    ctx.save_for_backward(x, y)
    
    return result
```
Before the `forward` method finishes, it takes `x` and `y` and shoves them into the `Context` backpack using `ctx.save_for_backward()`. This is crucial, because we will need those exact numbers later for calculus.

### 4. Rewrapping and Graph Building
Once `apply()` gets the raw `result` back from the `forward` method, it finishes its pipeline:
*   **Rewrapping**: It wraps the raw `result` back into a brand new `Tensor` (which will become `z`).
*   **Graph Building**: It creates a `GraphNode`. This node is a permanent record that says: *"This new Tensor was created by MultOp, its parents were x and y, and here is the Context bag with the saved variables."*
*   **Linking**: It attaches this `GraphNode` to `z`'s `_grad_fn` property. 

The forward pass is now complete. We have our answer (`z = 6.0`), and a perfectly constructed Computational Graph.

---

## Part 2: The Backward Pass (`z.backward()`)

Later in your code, you decide you want to find the derivatives. You call `z.backward()`. 

This wakes up the `AutogradEngine` (found in `gradience/autograd/autograd_engine.py`).

### 5. The Topological Sort
The engine looks at `z`'s `_grad_fn` (the GraphNode). It walks backward through the family tree and creates a flat, perfectly ordered list of every operation that happened. It ensures that parents are always processed after their children.

It then gives `z` a starting gradient of `1.0`. 

### 6. Passing the Gradient to `backward()`
The engine starts looping through the ordered list. It arrives at our `MultOp` node. 
It takes the incoming gradient (we call this `grad_output`) and calls `MultOp.backward(ctx, grad_output)`.

Notice that the engine passes the `ctx` (Context bag) back into the method!

### 7. Opening the Context Bag
Inside `MultOp.backward`, the very first thing we do is open the bag.
```python
@staticmethod
def backward(ctx, grad_output):
    # 1. Take x and y out of the bag
    x, y = ctx.saved_tensors
    
    # 2. Do the calculus
    grad_x = grad_output * y
    grad_y = grad_output * x
    ...
```
Because the `forward` method safely stored `x` and `y` in the bag earlier, the `backward` method can pull them out and use them to calculate the exact derivatives. 

### 8. The Secret Headache: Broadcasting
Before we return the gradients, we have to solve a math problem called **Broadcasting**.

Broadcasting is a convenience feature in NumPy. If you add a scalar (a single number) to a matrix (a grid of 10 numbers), NumPy secretly copies the scalar 10 times so the shapes match.

This is great for the forward pass, but it breaks the backward pass. The gradient that comes backward will be a grid of 10 numbers. If we try to save a grid of 10 gradients into a Tensor that only holds 1 number, Python will crash!

### 9. The Fix: `unbroadcast()`
To solve this, Gradience uses a special helper function called `unbroadcast()` (found in `gradience/utils/broadcast.py`). 

Before `MultOp.backward` returns its gradients, it passes them through `unbroadcast()`:
```python
    # 3. Unbroadcast the gradients back to their original shapes
    grad_x = unbroadcast(grad_x, x.shape)
    grad_y = unbroadcast(grad_y, y.shape)
    
    return grad_x, grad_y
```
The `unbroadcast()` function looks at the giant gradient matrix, compares it to the tiny original shape of the Tensor, and mathematically folds it back down until they match.

### 10. Gradient Accumulation
Finally, the `MultOp.backward` method spits out the perfectly shaped gradients. 

The Autograd Engine catches them and assigns them to the original `x` and `y` Tensors. But it does not overwrite them! It adds them. 
`tensor.grad = tensor.grad + new_grad`

This ensures that if `x` was used in multiple places in your math equation, none of its derivatives are lost. 

And that is it! You have now traced the exact code execution of a deep learning framework, from Python method interception, to the Context bag, to broadcasting, all the way to gradient accumulation!
