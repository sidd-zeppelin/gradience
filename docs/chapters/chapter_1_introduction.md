# Chapter 1: Introduction to Deep Learning

This chapter introduces the conceptual foundations of deep learning. We explain the transition from biological neurons to artificial neural structures. We derive the perceptron and multi-layer perceptrons. Finally, we discuss the universal approximation theorem.

## 1. Historical Motivation

In the middle of the twentieth century, researchers wanted to build machines that could learn. Early computer science relied on explicit programming rules. If a task was too complex to describe with manual rules, computers could not solve it. 

Researchers looked at the human brain for inspiration. The human brain consists of billions of interconnected cells called neurons. These biological cells process sensory input and allow learning. McCulloch and Pitts introduced the first mathematical model of a neuron in 1943. Rosenblatt expanded this model in 1958 by inventing the Perceptron. The Perceptron could learn weight parameters from data.

However, Minsky and Papert published a book in 1969 showing that single-layer perceptrons could not solve non-linear problems. Specifically, a single-layer perceptron could not learn the simple XOR logical function. This finding caused a decline in neural network research. 

To overcome this limitation, researchers added hidden layers of neurons. This led to Multi-Layer Perceptrons. With non-linear activation functions, these networks could approximate any continuous mathematical function.

## 2. Intuition

Imagine you want to decide whether to go to an outdoor concert. Your decision depends on three factors:
1. Is the weather good?
2. Is the ticket cheap?
3. Is your favorite band playing?

You assign a weight of importance to each factor. If the weather is vital to you, that factor gets a high weight. If you do not care about the ticket price, that factor gets a low weight. You sum the weighted factors. If the total exceeds a certain threshold, you decide to go.

An artificial neuron works the same way. It receives multiple inputs. It multiplies each input by a weight. It sums the results and adds a bias term. The bias shifts the activation threshold. Finally, an activation function decides the output value.

A single neuron can only draw a straight line to separate data. If the data classes are mixed in a non-linear way, a straight line is not enough. We combine multiple neurons into layers. The output of one layer becomes the input to the next layer. This network of layers can bend and warp the separation boundary to fit complex datasets.

## 3. Mathematical Foundations

Let $x \in \mathbb{R}^d$ be the input vector of features. Let $w \in \mathbb{R}^d$ be the weight vector. Let $b \in \mathbb{R}$ be the scalar bias.

The artificial neuron computes a weighted sum of its inputs and adds a bias:

$$z = \sum_{i=1}^d w_i x_i + b = w^T x + b$$

The perceptron output is determined by a step function $f(z)$:

$$y = f(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}$$

For the perceptron learning algorithm, we update the weights when the prediction is incorrect. Let $y$ be the true label and $\hat{y}$ be the predicted label. The update rule for weight $w_i$ and bias $b$ is:

$$w_i \leftarrow w_i + \eta (y - \hat{y}) x_i$$
$$b \leftarrow b + \eta (y - \hat{y})$$

where $\eta$ is the learning rate.

If the data is linearly separable, this algorithm will converge to a separating hyperplane. If the data is not linearly separable, the algorithm will loop indefinitely.

To model complex boundaries, we stack neurons. A Multi-Layer Perceptron (MLP) has an input layer, one or more hidden layers, and an output layer. Let $h^{(l)}$ be the activation vector of layer $l$. The equations for a layer are:

$$z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$$
$$h^{(l)} = \sigma(z^{(l)})$$

where $W^{(l)}$ is the weight matrix of layer $l$, $b^{(l)}$ is the bias vector, and $\sigma$ is a non-linear activation function.

The Universal Approximation Theorem states that a feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of $\mathbb{R}^d$, provided the activation function is non-linear and continuous.

## 4. Mathematical Intuition

Let us examine the neuron equation:

$$z = w^T x + b$$

The vector $w$ determines the orientation of the decision boundary. The boundary is a flat plane in high-dimensional space. The magnitude of $w$ determines the slope of the activation transition.

The bias $b$ determines the position of the decision boundary relative to the origin. If $b$ is positive, the boundary shifts away from the positive region. If $b$ is negative, it shifts towards it. Without the bias, the decision boundary would always pass through the coordinate origin. This would severely limit the model capability.

In an MLP, the weight matrix $W^{(l)}$ rotates and scales the feature space. The non-linear activation function $\sigma$ bends the space. Stacking these operations allows the network to isolate complex regions.

## 5. From Mathematics to Code

We map the mathematical equations to Python operations. 

Let the input $x$ be a batch of samples. We represent $x$ as a 2D array of shape `(batch_size, input_features)`. We represent the weight matrix $W$ as a 2D array of shape `(input_features, output_features)`. We represent the bias $b$ as a 1D array of shape `(output_features,)`.

The forward pass is:

$$Z = X W + b$$

In Python and NumPy, we implement this as:
```python
Z = np.dot(X, W) + b
```
NumPy automatically broadcasts the 1D bias vector $b$ across all rows of the 2D matrix multiplication result.

## 6. Gradience Implementation

In the Gradience framework, we implement this layer in the `Linear` class. The `Linear` class inherits from `Module`. It manages the weight and bias parameters as stateful `Parameter` objects.

Here is the implementation code from the framework:
```python
from gradience.tensor import Tensor
from gradience.nn.module import Module
from gradience.nn.parameter import Parameter
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        limit = np.sqrt(2.0 / in_features)
        weight_data = np.random.uniform(-limit, limit, (in_features, out_features))
        self.weight = Parameter(Tensor(weight_data))
        
        bias_data = np.zeros((out_features,))
        self.bias = Parameter(Tensor(bias_data))

    def forward(self, x):
        return x @ self.weight + self.bias
```
The constructor initializes weights using a uniform distribution. The range is scaled by the number of input features. The bias is initialized to zero. The forward method uses the matrix multiplication operator `@` of the `Tensor` class.

## 7. Complexity Analysis

Let $B$ be the batch size, $I$ be the number of input features, and $O$ be the number of output features.

* **Time Complexity**:
  * Forward Pass: $O(B \cdot I \cdot O)$ due to matrix multiplication.
  * Backward Pass: $O(B \cdot I \cdot O)$ to calculate gradients for inputs and weights.
* **Space Complexity**:
  * Parameter storage: $O(I \cdot O + O)$ for weights and biases.
  * Activation cache: $O(B \cdot O)$ to store outputs for the backward pass.

## 8. Visualizations

Here is a diagram of a single artificial neuron:

```
Inputs      Weights        Sum & Bias        Activation
 x1 -------> w1 ------\
                       \
 x2 -------> w2 --------+---> [ Sum ] + b ---> [ Function ] ---> Output
                       /
 x3 -------> w3 ------/
```

Here is the computational graph of the linear layer:

```
  x (Tensor)        w (Parameter)
     \               /
      \             /
       v           v
       [ MatMulOp ]        b (Parameter)
            \               /
             \             /
              v           v
              [ AddOp ]
                 |
                 v
              Output
```

## 9. Comparisons

| Model | Decision Boundary | XOR Capability | Learning Rule |
| :--- | :--- | :--- | :--- |
| Perceptron | Linear | No | Perceptron error correction |
| Multi-Layer Perceptron | Non-linear | Yes | Backpropagation |

Single perceptrons are computationally cheap but cannot solve complex classification problems. MLPs are expressive but require gradient descent optimization.

## 10. Practical Applications

Neural network layers are the building blocks of deep learning. Fully connected linear layers are used in:
* The final classification heads of Convolutional Neural Networks.
* Feedforward sublayers in Transformer encoders and decoders.
* Policy and value networks in reinforcement learning models.

## 11. Common Mistakes

* **Zero Initialization**: Initializing all weights in an MLP to zero. This causes all hidden units to compute identical gradients. The model cannot break symmetry.
* **Forgetting the Bias**: Omitting the bias term. This forces the decision boundary to pass through the origin.
* **Using Linear Activations**: Stacking multiple linear layers without non-linear activations. A composition of linear layers is mathematically equivalent to a single linear layer.

## 12. Exercises

### Conceptual Questions
1. Why does a composition of linear layers fail to increase the representational power of a network?
2. Explain how a bias term shifts the decision boundary of a neuron.

### Mathematical Exercises
1. Prove that the XOR function cannot be solved by a single-layer perceptron.
2. Show that any network with linear activation functions can be compressed into a single linear layer.

### Programming Exercises
1. Implement a perceptron algorithm from scratch using NumPy. Test it on a linearly separable binary dataset.

### Debugging Exercises
1. A network has all weights initialized to the constant 5.0. It fails to learn. Explain why and fix the initialization.

### Research Questions
1. How does the choice of weight initialization scale with the number of layers in deep networks?

## 13. References

* McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. *Bulletin of Mathematical Biophysics*, 5(4), 115-133.
* Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386.
* Minsky, M., & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press.
