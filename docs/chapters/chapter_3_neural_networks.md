# Chapter 3: Neural Networks

This chapter describes the components of feedforward neural networks. We define linear layers. We derive the activations and loss functions. We describe forward and backward propagation. Finally, we build Multi-Layer Perceptrons.

## 1. Historical Motivation

Early neural network designs struggled with training stability. While Rosenblatt's perceptron could learn, it was limited to linear functions. When researchers added multiple layers, they needed a way to optimize the hidden parameter representations.

Computing output gradients with respect to weights in deep layers requires a systematic propagation rule. Simple numerical evaluations of gradients do not scale. Manually writing the derivatives for every unique architecture is too slow. Stacking layer classes with structured forward and backward interfaces solves this.

Additionally, layers require non-linear activations to learn complex boundaries. Early models used step functions, but step functions are non-differentiable at zero and have zero derivatives everywhere else. They cannot propagate gradients. Smooth activations like Sigmoid and Tanh solved this, followed by ReLU to prevent gradient saturation.

## 2. Intuition

A neural network is a chain of mathematical transformations. Each layer performs a specific task.

First, the linear layer acts as a projection. It rotates, stretches, or squeezes the input data space.

Second, the activation layer bends the data space. Without this step, no matter how many layers you stack, the network remains a single linear projection. Stacking linear and activation layers is like folding a sheet of paper. You fold and bend the paper until a single straight cut can separate the complex patterns.

Third, the loss layer measures the network performance. It outputs a single error score. The backward pass pushes this error back through the folds. Each layer adjusts its weights to reduce the error on the next iteration.

## 3. Mathematical Foundations

Let $x \in \mathbb{R}^{B \times I}$ be the input batch, where $B$ is the batch size and $I$ is the number of input features.

### 1. Linear Layer

The forward mapping is:

$$Z = X W + b$$

where $W \in \mathbb{R}^{I \times O}$ is the weight matrix and $b \in \mathbb{R}^O$ is the bias vector.
During backpropagation, we receive the incoming gradient $\frac{\partial L}{\partial Z} \in \mathbb{R}^{B \times O}$. The gradients with respect to inputs, weights, and biases are:

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} W^T$$
$$\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Z}$$
$$\frac{\partial L}{\partial b} = \sum_{i=1}^B \frac{\partial L}{\partial Z_i}$$

### 2. Activation Functions

* **Sigmoid**:
  
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
  
  Local derivative:
  
  $$\sigma'(z) = \sigma(z) (1 - \sigma(z))$$

* **Tanh**:
  
  $$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
  
  Local derivative:
  
  $$\tanh'(z) = 1 - \tanh^2(z)$$

* **ReLU**:
  
  $$\text{ReLU}(z) = \max(0, z)$$
  
  Local derivative:
  
  $$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

### 3. Loss Functions

* **Mean Squared Error (MSE)**:
  
  $$\text{MSE}(y, \hat{y}) = \frac{1}{N \cdot M} \sum_{i=1}^N \sum_{j=1}^M (y_{ij} - \hat{y}_{ij})^2$$
  
  Gradient with respect to predictions $\hat{y}$:
  
  $$\frac{\partial \text{MSE}}{\partial \hat{y}_{ij}} = \frac{2}{N \cdot M} (\hat{y}_{ij} - y_{ij})$$

* **Cross-Entropy Loss (with Softmax)**:
  Let $s_i$ be the raw prediction score (logit) for class $i$. The softmax probability is:
  
  $$p_i = \frac{e^{s_i}}{\sum_j e^{s_j}}$$
  
  The Cross-Entropy Loss for a single label $y$ (one-hot vector) is:
  
  $$L = -\sum_i y_i \log(p_i)$$
  
  The analytical gradient with respect to logit $s_i$ is:
  
  $$\frac{\partial L}{\partial s_i} = p_i - y_i$$

## 4. Mathematical Intuition

Let us study the Cross-Entropy gradient:

$$\frac{\partial L}{\partial s_i} = p_i - y_i$$

The term $p_i$ is the probability the network assigns to class $i$. The term $y_i$ is the target label.

If the network assigns a high probability $p_i \approx 1$ to the correct class $y_i = 1$, the gradient $p_i - y_i \approx 0$. No update is made.

If the network assigns a low probability $p_i \approx 0.1$ to the correct class $y_i = 1$, the gradient is negative: $0.1 - 1 = -0.9$. This negative gradient pushes the logit value up.

If the network assigns a high probability $p_i \approx 0.8$ to an incorrect class $y_i = 0$, the gradient is positive: $0.8 - 0 = 0.8$. This positive gradient pushes the logit value down.

## 5. From Mathematics to Code

We implement the layers by subclassing the autograd `Function`. This guarantees that they integrate with the tape-based backpropagation engine.

Let us map the Tanh activation.
Mathematics:

$$a = \tanh(z)$$
$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} (1 - a^2)$$

Code implementation:
```python
class TanhOp(Function):
    @staticmethod
    def forward(ctx, x):
        out = np.tanh(x.data)
        ctx.save_for_backward(out)
        return Tensor(out)

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.saved_tensors[0]
        return grad_output * (1.0 - out ** 2)
```

We cache the output value `out` in the context during the forward pass. We use it directly in the backward pass.

## 6. Gradience Implementation

The stateful `Module` container coordinates the forward pass execution and groups the model parameters. 

Here is the implementation of the `Sequential` container from the framework:
```python
class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```
The constructor adds each layer as a submodule. This registers its parameters recursively. The forward method loops over the layers, piping the output of each layer into the next one.

## 7. Complexity Analysis

Let $B$ be the batch size, $I$ be the input features, and $O$ be the output features.

* **Time Complexity**:
  * Linear Layer: $O(B \cdot I \cdot O)$ for forward and backward passes.
  * Activations: $O(B \cdot O)$ since they are element-wise.
  * Cross-Entropy: $O(B \cdot O)$ to compute softmax and loss.
* **Space Complexity**:
  * Parameters: $O(I \cdot O)$ for weights and $O(O)$ for biases.
  * Cached outputs: $O(B \cdot O)$ to compute backward steps.

## 8. Visualizations

Here is a visual diagram of a Multi-Layer Perceptron (MLP):

```
Inputs (x)        Hidden Layer (h)       Output Layer (y)
   ( )  ---------\     ( )  ----------\        ( )
                  \                    \
   ( )  -----------+-> ( )  ------------+----> ( )
                  /                    /
   ( )  ---------/     ( )  ----------/        ( )
```

Here is the computational flow of the forward pass:

```
  X ---> [ Linear Layer ] ---> Z ---> [ Activation ] ---> H ---> Output
```

## 9. Comparisons

| Activation | Range | Gradients | Computational Cost |
| :--- | :--- | :--- | :--- |
| Sigmoid | $(0, 1)$ | Saturation at extremes | High (Exponential) |
| Tanh | $(-1, 1)$ | Saturation at extremes | High (Exponential) |
| ReLU | $[0, \infty)$ | Constant gradient if $> 0$ | Extremely Low |

## 10. Practical Applications

Neural networks are used across all machine learning fields:
* **MLPs**: Simple classification and regression tasks.
* **Activations**: ReLU is used in deep convolutional networks to speed up training. Softmax is used at the output layer for multi-class classification.

## 11. Common Mistakes

* **Vanishing Gradients**: Stacking many layers with Sigmoid or Tanh activations. The gradients saturate at the extremes, becoming close to zero. This halts weight optimization.
* **Dying ReLU**: Large gradients updating weights such that a ReLU unit never activates. The derivative remains zero forever.
* **Loss Mismatch**: Using Cross-Entropy on logits that have already passed through a softmax activation. This applies the logarithm twice.

## 12. Exercises

### Conceptual Questions
1. Why does Tanh usually perform better than Sigmoid in hidden layers?
2. Explain the dying ReLU problem and how to mitigate it.

### Mathematical Exercises
1. Derive the derivative of the Softmax function with respect to its input scores.
2. Prove that the derivative of the Sigmoid function can be expressed purely in terms of its output value.

### Programming Exercises
1. Implement the `Softmax` function in NumPy. Handle numerical instability when input values are very large.

### Debugging Exercises
1. A training loop uses `CrossEntropyLoss` but applies `Sigmoid` to the network outputs first. Fix the forward architecture.

### Research Questions
1. How do activations like ELU or GELU address the limitations of standard ReLU?

## 13. References

* Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restriction Boltzmann machines. *Proceedings of ICML*.
* Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep sparse rectifier neural networks. *Proceedings of AISTATS*.
* Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
