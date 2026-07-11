# Chapter 2: Building the Foundations

This chapter outlines the core mathematical and computational foundations of automatic differentiation. We define tensors. We build computational graphs. We derive reverse-mode automatic differentiation. Finally, we implement our autograd engine.

## 1. Historical Motivation

In the early days of machine learning, training neural networks required manually deriving and coding gradients. For a simple network, this was feasible. However, as networks grew in depth and complexity, manual derivation became error-prone and tedious. A small change in layer configuration required recalculating all gradient derivations.

Symbolic differentiation tools like Mathematica could compute exact analytical derivatives. However, they suffered from expression swelling. The derivative equations grew exponentially with network depth. Numerical differentiation using finite differences was simple to implement. However, it was computationally expensive. It required running a full forward pass of the model for every single parameter. For models with millions of parameters, this was unusable.

Automatic differentiation (Autograd) was developed to solve this. Automatic differentiation is not symbolic nor numerical. It computes derivatives by applying the mathematical chain rule to a sequence of elementary operations. Reverse-mode automatic differentiation is particularly powerful. It computes the gradients of a scalar objective with respect to all input parameters in a single backward pass. This enables efficient backpropagation in deep neural networks.

## 2. Intuition

Imagine you are building a factory assembly line. Raw materials go through a sequence of machines. Each machine performs one basic operation. For example, machine A adds two inputs. Machine B multiplies two inputs. The final product is a single number representing the factory score.

If you want to know how a change in raw materials affects the final score, you can trace the assembly line. During the forward pass, materials flow forward. We record the operation performed by each machine.

During the backward pass, we trace the assembly line in reverse. We start at the end of the line. The final machine knows how its output changes relative to its inputs. It passes this sensitivity score backward to the previous machines. Each machine multiplies the incoming sensitivity score by its own local derivative. This backward flow is the chain rule in action. By the time we reach the start of the line, we have the sensitivity score for every raw material.

A tensor is the raw material. The computational graph is the assembly line. Reverse-mode automatic differentiation is the backward flow of sensitivity scores.

## 3. Mathematical Foundations

Let $f: \mathbb{R}^n \to \mathbb{R}$ be a composite function defined as a sequence of elementary operations. Let $v_i$ represent the intermediate variables computed during evaluation.

For a node $v_i$ with parent nodes parents($v_i$), the forward pass computes:

$$v_i = \phi_i(\{ v_j \}_{j \in \text{parents}(v_i)})$$

Let $y$ be the final scalar output of the function. We define the adjoint (or gradient) of variable $v_i$ as:

$$\bar{v}_i = \frac{\partial y}{\partial v_i}$$

By the chain rule of calculus, the adjoint of a variable $v_j$ is the sum of the adjoints of its children nodes, weighted by the local partial derivatives:

$$\bar{v}_j = \sum_{i \in \text{children}(v_j)} \bar{v}_i \frac{\partial v_i}{\partial v_j}$$

In reverse-mode automatic differentiation, we perform two passes:
1. **Forward Pass**: Evaluate the operations sequentially to compute the outputs and store intermediate values.
2. **Backward Pass**: Initialize the final output adjoint $\bar{y} = 1.0$. Traverse the computational graph in reverse topological order. Compute the adjoints of all intermediate nodes.

Topological sorting guarantees that we only compute the adjoint $\bar{v}_j$ after we have computed the adjoints of all its children nodes. A directed acyclic graph (DAG) always has at least one topological ordering.

## 4. Mathematical Intuition

Let us examine the adjoint equation:

$$\bar{v}_j = \bar{v}_i \frac{\partial v_i}{\partial v_j}$$

The term $\frac{\partial v_i}{\partial v_j}$ represents the local derivative of the operation. It measures how the node output changes relative to its input in isolation.

The term $\bar{v}_i$ represents the global sensitivity of the final output with respect to the node output. It measures how the final loss changes if we nudge the intermediate node.

Multiplying these two terms propagates the global sensitivity one step backward. This separates local operation mechanics from global network structures. A node only needs to know how to differentiate its immediate operation. It does not need to know the structure of the rest of the network.

## 5. From Mathematics to Code

We translate the mathematical computational graph into Python code.

A `Tensor` represents a node in the graph. It wraps a NumPy array and stores its gradient. It also holds references to its parent nodes and the operation that created it.

Let the operation be addition: $z = x + y$.
The local derivatives are:

$$\frac{\partial z}{\partial x} = 1, \quad \frac{\partial z}{\partial y} = 1$$

The backward step updates the gradients of the inputs:

$$\bar{x} \leftarrow \bar{x} + \bar{z}$$
$$\bar{y} \leftarrow \bar{y} + \bar{z}$$

In Python, we code this backward step as:
```python
def backward(grad_output):
    grad_x = grad_output
    grad_y = grad_output
    return grad_x, grad_y
```

## 6. Gradience Implementation

Here is the implementation of the autograd core. It shows the `Function` class and the topological sort algorithm.

```python
class GraphNode:
    def __init__(self, tensor):
        self.tensor = tensor
        self.parents = []
        self.creator = None

class Function:
    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

class AutogradEngine:
    @staticmethod
    def backward(target_tensor):
        if target_tensor.grad is None:
            target_tensor.grad = np.ones_like(target_tensor.data)
            
        topo_order = []
        visited = set()
        
        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                if tensor.creator is not None:
                    for parent in tensor.creator.parents:
                        build_topo(parent)
                topo_order.append(tensor)
                
        build_topo(target_tensor)
        
        for tensor in reversed(topo_order):
            if tensor.creator is None:
                continue
            grads = tensor.creator.backward(tensor.grad)
            if not isinstance(grads, tuple):
                grads = (grads,)
            for parent, grad in zip(tensor.creator.parents, grads):
                if parent.requires_grad:
                    if parent.grad is None:
                        parent.grad = np.array(grad)
                    else:
                        parent.grad = parent.grad + np.array(grad)
```

The `build_topo` helper executes a depth-first search (DFS) to build the topological sort. The engine then processes the nodes in reverse order, accumulating gradients into `parent.grad`.

## 7. Complexity Analysis

Let $V$ be the number of variables (nodes) and $E$ be the number of operations (edges) in the computational graph.

* **Time Complexity**:
  * Forward Pass: $O(V + E)$ since each node is evaluated once.
  * Topological Sort: $O(V + E)$ using depth-first search.
  * Backward Pass: $O(V + E)$ since each local derivative is computed once.
* **Space Complexity**:
  * Graph storage: $O(V + E)$ to store the nodes and parent pointers.
  * Activation storage: $O(V)$ to store intermediate arrays for backpropagation.

## 8. Visualizations

Here is a computational graph showing branching and gradient accumulation:

```
    a (Input)
     \
      v
  [ Square ] ----> b (Intermediate)
     /      \
    /        \
   v          v
[ Add ]    [ Multiply ]
   \          /
    v        v
    c (Scalar Loss)
```

During backward pass on node $b$:

$$\bar{b} = \bar{c}_{\text{left}} \cdot 1.0 + \bar{c}_{\text{right}} \cdot a$$

Accumulating the gradients avoids the multi-path differentiation bug.

## 9. Comparisons

| Method | Time Complexity (Parameters $N$) | Accuracy | Graph Overhead |
| :--- | :--- | :--- | :--- |
| Numerical (Finite Differences) | $O(N)$ forward passes | Approximate | None |
| Symbolic Differentiation | Expression swelling | Exact | High |
| Reverse-Mode Autograd | $O(1)$ backward pass | Exact | Low (Dynamic graph) |

## 10. Practical Applications

Reverse-mode autograd is used in all modern deep learning pipelines. It allows:
* Optimizing weights in convolutional neural networks.
* Training large transformer models with billions of parameters.
* Tuning hyperparameters via gradient updates.

## 11. Common Mistakes

* **In-place Modifications**: Changing tensor values in-place during the forward pass. This overwrites cached activations needed for the backward pass.
* **Forgetting gradient accumulation**: Overwriting gradients instead of summing them when a node has multiple children. This leads to incorrect gradient values.
* **Retaining Graphs**: Not clearing references to the computational graph after the backward pass. This causes memory leaks.

## 12. Exercises

### Conceptual Questions
1. What is the difference between forward-mode and reverse-mode automatic differentiation? Why is reverse-mode preferred for neural networks?
2. Explain why topological sorting is required before running the backward pass.

### Mathematical Exercises
1. Trace the forward and backward values for the function $y = (x_1 \cdot x_2) + \sin(x_1)$ at $x_1 = 2.0, x_2 = 3.0$.
2. Derive the local partial derivatives for the matrix multiplication operation $Z = X W$.

### Programming Exercises
1. Write a custom `Function` subclass for the division operation $z = x / y$. Implement both forward and backward methods.

### Debugging Exercises
1. A node gradient is calculated as double the expected value. The node branches to two children. Diagnose the issue and fix the gradient accumulation.

### Research Questions
1. How do modern deep learning frameworks reduce memory overhead by recomputing activations during the backward pass?

## 13. References

* Wengert, R. E. (1964). A simple automatic derivative evaluation program. *Communications of the ACM*, 7(8), 463-464.
* Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
* Griewank, A., & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*. SIAM.
