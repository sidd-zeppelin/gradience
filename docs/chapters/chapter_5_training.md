# Chapter 5: Training Neural Networks

This chapter describes the techniques required to train stable and generalized deep networks. We cover datasets and dataloaders. We analyze parameter weight initialization. We derive normalization layers. Finally, we formulate regularization via dropout.

## 1. Historical Motivation

As neural networks grew deeper, training became highly unstable. First, researchers observed the vanishing and exploding gradient problems. If weights were initialized too small, activations decayed exponentially as they passed through layers, resulting in zero gradients. If weights were initialized too large, activations exploded, leading to numerical overflow (NaN values). Glorot & Bengio (2010) and He et al. (2015) solved this by deriving variance-scaling initializers based on the number of inputs and outputs of a layer.

Second, the distribution of intermediate activations changed constantly during training as parameters updated. This is called internal covariate shift. It forced later layers to constantly adapt to changing inputs, slowing down training. Ioffe & Szegedy (2015) solved this by introducing Batch Normalization, which normalizes activations using batch statistics. Ba et al. (2016) introduced Layer Normalization to stabilize recurrent networks and handle variable sequence lengths.

Third, deep models easily memorized training data, leading to overfitting. The model performed well on training sets but failed on unseen test sets. Srivastava et al. (2014) introduced Dropout, which randomly disables neurons during training to prevent co-adaptation of features.

## 2. Intuition

* **Weight Initialization**: Imagine a megaphone chain. If each person whispers slightly quieter than the last, the message is lost at the end. If each person shouts louder than the last, the volume deafens the final listener. We must scale the voice volume of each person to exactly match their input. This is what Xavier and He initializations do. They scale weight variance so activation signals remain constant across layers.
* **Normalization**: Imagine a class of students writing essays. If the grading scale changes daily, students cannot improve. Batch Normalization scales student marks to have a mean of zero and variance of one. This ensures constant activation distributions, stabilizing learning.
* **Dropout**: Imagine a soccer team. If the team relies entirely on one star player, the team fails when that player is injured. If you randomly bench players during practice, the remaining teammates must learn to coordinate without relying on a single player. Dropout randomly disables connections, forcing the network to learn robust, redundant representations.

## 3. Mathematical Foundations

### 1. Weight Initialization

Let $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ be a weight matrix. We calculate the fan-in $d_{\text{in}}$ (number of input units) and fan-out $d_{\text{out}}$ (number of output units).

* **Xavier (Glorot) Uniform**:
  Initialize weights from a uniform distribution:
  
  $$W_{ij} \sim U(-a, a) \quad \text{where } a = \sqrt{\frac{6}{d_{\text{in}} + d_{\text{out}}}}$$

* **He (Kaiming) Normal**:
  Initialize weights from a normal distribution:
  
  $$W_{ij} \sim N(0, \sigma^2) \quad \text{where } \sigma = \sqrt{\frac{2}{d_{\text{in}}}}$$

### 2. Batch Normalization (BatchNorm1d)

Let $x \in \mathbb{R}^{B \times C}$ be a batch of features, where $B$ is the batch size and $C$ is the channel dimension. We compute the mean and variance across the batch dimension $B$ for each channel:

$$\mu_c = \frac{1}{B} \sum_{i=1}^B x_{ic}$$
$$\sigma_c^2 = \frac{1}{B} \sum_{i=1}^B (x_{ic} - \mu_c)^2$$

We normalize the activations:

$$\hat{x}_{ic} = \frac{x_{ic} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}$$

We scale and shift the normalized values using learnable parameters $\gamma$ and $\beta$:

$$y_{ic} = \gamma_c \hat{x}_{ic} + \beta_c$$

During evaluation (eval mode), we use running averages of mean and variance accumulated during training:

$$\text{running\_mean} \leftarrow (1 - \text{momentum}) \cdot \text{running\_mean} + \text{momentum} \cdot \mu$$
$$\text{running\_var} \leftarrow (1 - \text{momentum}) \cdot \text{running\_var} + \text{momentum} \cdot \sigma^2$$

### 3. Layer Normalization (LayerNorm)

We compute the mean and variance across the channel dimension $C$ for each individual sample $i$:

$$\mu_i = \frac{1}{C} \sum_{c=1}^C x_{ic}$$
$$\sigma_i^2 = \frac{1}{C} \sum_{c=1}^C (x_{ic} - \mu_i)^2$$

We normalize, scale, and shift:

$$\hat{x}_{ic} = \frac{x_{ic} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}$$
$$y_{ic} = \gamma_c \hat{x}_{ic} + \beta_c$$

Unlike BatchNorm, LayerNorm performs identical operations during training and evaluation.

### 4. Dropout

During training, we generate a binary mask $M_{ic}$ where each element is drawn from a Bernoulli distribution:

$$M_{ic} \sim \text{Bernoulli}(1 - p)$$

where $p$ is the dropout probability. We scale the active connections to maintain expected activation magnitudes:

$$y_{ic} = \frac{M_{ic}}{1 - p} \cdot x_{ic}$$

During evaluation (eval mode), dropout is deactivated: $y = x$.

## 4. Mathematical Intuition

Let us examine the scaling factor in Dropout:

$$y = \frac{M}{1 - p} \cdot x$$

If the dropout rate $p = 0.5$, half of the neurons are deactivated on average. The sum of activations entering the next layer would drop by half.

To prevent this magnitude shift, we divide the active outputs by $1 - p = 0.5$. This scales the remaining activations by $2.0$. The expected sum of activations remains constant. This is called inverted dropout. It ensures we do not modify the network parameters during evaluation mode.

## 5. From Mathematics to Code

We translate the mathematical operations into Python.

Let us map the Kaiming variance calculation.
For a 4D convolutional weight tensor of shape `(out_channels, in_channels, kernel_height, kernel_width)`, the fan-in is:

$$d_{\text{in}} = \text{in\_channels} \cdot \text{kernel\_height} \cdot \text{kernel\_width}$$

Code implementation:
```python
def calculate_fan(shape):
    if len(shape) == 2:
        return shape[0], shape[1]
    elif len(shape) == 4:
        receptive_field = shape[2] * shape[3]
        return shape[1] * receptive_field, shape[0] * receptive_field
```

Let us map the Dropout forward pass.
Code implementation:
```python
# During training:
mask = (np.random.rand(*x.shape) >= p) / (1.0 - p)
y = x * mask
```

## 6. Gradience Implementation

Here is the implementation of `BatchNorm1d` from the framework. It manages parameters, running averages, and mode toggles.

```python
class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(Tensor(np.ones((num_features,))))
        self.beta = Parameter(Tensor(np.zeros((num_features,))))
        self.running_mean = np.zeros((num_features,))
        self.running_var = np.ones((num_features,))

    def forward(self, x):
        if self.training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var.data
            
            x_norm = (x - mean) / ((var + self.eps) ** 0.5)
        else:
            x_norm = (x - self.running_mean) / ((self.running_var + self.eps) ** 0.5)
            
        return x_norm * self.gamma + self.beta
```
The forward method branches based on `self.training`. It updates running statistics during training, and uses them during evaluation.

## 7. Complexity Analysis

Let $B$ be the batch size and $C$ be the number of features.

* **Time Complexity**:
  * BatchNorm1d / LayerNorm: $O(B \cdot C)$ for forward and backward passes.
  * Dropout: $O(B \cdot C)$ to generate the mask.
* **Space Complexity**:
  * Parameter storage: $O(C)$ for $\gamma$ and $\beta$.
  * Cache size: $O(B \cdot C)$ to store normalized activations.

## 8. Visualizations

Here is a visual comparison of normalization directions:

```
Batch Normalization (BatchNorm1d)       Layer Normalization (LayerNorm)
     Batch Dimension (B)                     Batch Dimension (B)
     +---------------+                       +---------------+
     |   |   |   |   |                       |====== mean ===| -> Sample 1
     |   v   v   v   v                       |====== mean ===| -> Sample 2
     | mean per col  |                       |====== mean ===| -> Sample 3
     +---------------+                       +---------------+
       Channel (C)                             Channel (C)
```

## 9. Comparisons

| Layer | Normalization Axis | Dependency on Batch Size | Primary Use Case |
| :--- | :--- | :--- | :--- |
| BatchNorm1d | Batch Dimension | High (Fails for small batches) | Convolutional Neural Networks |
| LayerNorm | Channel Dimension | None | Recurrent Networks & Transformers |

## 10. Practical Applications

These training layers are standard in modern deep neural networks:
* **Batch Normalization**: Used in deep vision networks like ResNet to accelerate training.
* **Layer Normalization**: Used in Transformer self-attention blocks.
* **Dropout**: Used in the classification heads of MLPs and AlexNet to prevent overfitting.

## 11. Common Mistakes

* **Dataloader Shuffling**: Forgetting to shuffle the training dataset. This leads to biased gradient estimates.
* **Evaluation Mode**: Forgetting to call `.eval()` on the module before running inference. This keeps Dropout active and updates running statistics in BatchNorm.
* **Mismatch in Normalization**: Using BatchNorm with a batch size of 1. The variance becomes zero, leading to division by zero or NaN values.

## 12. Exercises

### Conceptual Questions
1. Why does Batch Normalization fail when the batch size is very small? How does Layer Normalization resolve this?
2. Explain the training and evaluation differences for a Dropout layer.

### Mathematical Exercises
1. Derive the gradients for Layer Normalization with respect to the input activations.
2. Calculate the variance of the outputs of a linear layer $y = Wx$ under He initialization, assuming the inputs have unit variance.

### Programming Exercises
1. Implement the `LayerNorm` class in Gradience. Test its output against the PyTorch equivalent.

### Debugging Exercises
1. A model trained with Dropout outputs erratic predictions during evaluation. Diagnose the issue and fix the code.

### Research Questions
1. How does Weight Standardization stabilize training when combined with Group Normalization?

## 13. References

* Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *Proceedings of AISTATS*.
* Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929-1958.
* Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *Proceedings of ICML*.
* He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *Proceedings of ICCV*.
* Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. *arXiv preprint arXiv:1607.06450*.
