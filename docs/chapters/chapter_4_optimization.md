# Chapter 4: Optimization

This chapter details the mathematical optimization algorithms used to train neural networks. We describe gradient descent variants. We derive standard momentum, Nesterov acceleration, and weight decay. We analyze adaptive learning rate algorithms. Finally, we formulate learning rate decay schedules.

## 1. Historical Motivation

Stochastic Gradient Descent (SGD) updates parameters by moving them opposite the gradient direction. While simple and computationally cheap, basic SGD has limitations. First, if the loss surface is highly non-spherical (such as a narrow valley), SGD will oscillate back and forth across the valley walls instead of moving down the valley floor. Second, SGD struggles to escape saddle points or flat plateaus where gradients are close to zero.

To accelerate training, researchers added momentum. Momentum mimics a physical ball rolling down a hill. It accumulates velocity from past gradients to damp oscillations and speed up descent. Nesterov momentum improved this by calculating gradients at a look-ahead position.

However, a single global learning rate is not ideal for all parameters. Sparse features require larger updates than frequent features. Duchi et al. solved this by introducing AdaGrad, which scales the learning rate of each parameter based on the sum of squares of its historical gradients. AdaGrad, however, suffers from a decaying learning rate that eventually becomes zero, stopping learning. RMSprop solved this by using an exponential moving average of squared gradients. Adam combined momentum and RMSprop into a unified optimizer. AdamW corrected how weight decay is applied in Adam.

## 2. Intuition

Imagine you are trying to find the lowest point in a hilly terrain during a heavy fog. You can only feel the slope of the ground under your feet.

* **Basic SGD**: You take a step down the steepest slope. If you are in a steep ravine, you bounce from one side of the ravine to the other. Your progress along the ravine floor is slow.
* **Momentum**: You gain velocity as you move down. The velocity carries you across small bumps and prevents you from oscillating sideways.
* **Nesterov Momentum**: You look ahead along your current path. If you see the ground is about to rise, you slow down early to prevent overshooting the valley bottom.
* **AdaGrad / RMSprop**: You adjust your step size for each coordinate direction. If a coordinate direction has had massive slopes, you take very small, cautious steps in that direction. If a coordinate direction has had flat slopes, you take larger steps in that direction to explore.
* **Adam**: You combine momentum (velocity tracking) and RMSprop (step size scaling) to navigate the terrain efficiently.

## 3. Mathematical Foundations

Let $\theta_t$ represent the parameters of the model at step $t$. Let $g_t = \nabla_\theta L(\theta_t)$ be the gradient computed on the current batch. Let $\eta$ be the base learning rate.

### 1. Stochastic Gradient Descent (SGD) with Momentum and Weight Decay

Standard parameter update:

$$\theta_{t+1} = \theta_t - \eta g_t$$

With weight decay parameter $\lambda$:

$$g_t \leftarrow g_t + \lambda \theta_t$$

With momentum factor $\beta$ and velocity $v_t$:

$$v_{t+1} = \beta v_t + g_t$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

With Nesterov acceleration, we evaluate the gradient at a look-ahead position $\theta_t - \beta v_t$:

$$v_{t+1} = \beta v_t + \nabla_\theta L(\theta_t - \beta v_t)$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

### 2. AdaGrad

We accumulate the sum of squares of all historical gradients:

$$G_t = G_{t-1} + g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t$$

where $\epsilon$ is a small constant to prevent division by zero.

### 3. RMSprop

We replace the simple sum with an exponential moving average of squared gradients:

$$v_t = \beta v_{t-1} + (1 - \beta) g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} g_t$$

### 4. Adam & AdamW

We track both the first raw moment $m_t$ (momentum) and the second raw moment $v_t$ (RMSprop):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Because $m_0$ and $v_0$ are initialized to zero, they are biased toward zero. We apply bias correction terms:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

The Adam update rule is:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

In AdamW, weight decay $\lambda$ is applied directly to the parameters rather than adding $\lambda \theta_t$ to the gradient $g_t$:

$$\theta_{t+1} = \theta_t - \eta \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

### 5. Learning Rate Decay

Learning rate schedulers reduce the learning rate over time:
* **Step Decay**: Multiply $\eta$ by a factor $\gamma$ every $S$ steps.
* **Exponential Decay**:
  
  $$\eta_t = \eta_0 e^{-k \cdot t}$$

## 4. Mathematical Intuition

Let us examine the Adam update rule:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

The term $\hat{m}_t$ determines the direction of the step. It is a smoothed version of the gradient. It carries the parameter along consistent directions, damping high-frequency noise.

The term $\sqrt{\hat{v}_t} + \epsilon$ determines the scale of the step. If a parameter has had small gradients, $\hat{v}_t$ is small. This increases the step size. If a parameter has had large gradients, $\hat{v}_t$ is large. This decreases the step size. This bounds the effective step size, stabilizing training.

In Adam, adding weight decay to the gradient $g_t$ causes parameters with large historical gradients to decay slower than parameters with small historical gradients. AdamW decouple weight decay from gradient scaling. This ensures all parameters decay at the same rate.

## 5. From Mathematics to Code

We implement the optimizer update loop by iterating over the registered `Parameter` list.

Let us map the RMSprop update.
Mathematics:

$$v \leftarrow \beta v + (1 - \beta) g^2$$
$$\theta \leftarrow \theta - \frac{\eta}{\sqrt{v} + \epsilon} g$$

Code implementation:
```python
# For each parameter p in the list of parameters:
# We fetch its gradient p.grad and the cached velocity v
v_param = beta * v_param + (1.0 - beta) * (p.grad ** 2)
p.data = p.data - (lr / (np.sqrt(v_param) + eps)) * p.grad
```
We store the velocity cache `v_param` as a NumPy array of the same shape as `p.data`.

## 6. Gradience Implementation

Here is the implementation of the `Adam` optimizer class from the framework. It implements parameter state tracking and the bias-corrected update rules.

```python
class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            
            p.data = p.data - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat
```
The optimizer stores state vectors `m` and `v` for each parameter. The `step` method increments the step counter `self.t` and evaluates the updates element-wise.

## 7. Complexity Analysis

Let $P$ be the total number of parameters in the network.

* **Time Complexity**:
  * Step update: $O(P)$ since we perform element-wise arithmetic on each parameter vector.
* **Space Complexity**:
  * SGD (no momentum): $O(1)$ extra space.
  * SGD with Momentum / RMSprop: $O(P)$ to store one state vector.
  * Adam / AdamW: $O(2P)$ to store two state vectors (first and second moments).

## 8. Visualizations

Here is a diagram comparing optimization paths on a ravine loss surface:

```
ravine wall
-----------
  \     /
   \   /      SGD (oscillates heavily)
    \ /
     v
     |
     |        Momentum (dampens oscillations, moves down)
     v
-----------
ravine floor
```

Here is the computational dependency of the Adam update step:

```
  g_t (Gradient) -------> m_t (First Moment) ------> m_hat
                   \                                         \
                    v                                         v
                  g_t^2 --> v_t (Second Moment) -----> v_hat --> [ Update Equation ] ---> Parameter (theta)
```

## 9. Comparisons

| Optimizer | Memory Overhead | Handles Sparse Gradients | Tuning Complexity |
| :--- | :--- | :--- | :--- |
| SGD | None | Poorly | High (Learning rate) |
| SGD + Momentum | Low ($O(P)$) | Moderately | High (Learning rate, momentum) |
| RMSprop | Low ($O(P)$) | Well | Medium (Learning rate, decay rate) |
| Adam | High ($O(2P)$) | Well | Low (Robust default hyperparameters) |

## 10. Practical Applications

Choosing an optimizer depends on the model architecture:
* **SGD with Momentum**: Often used in ResNet CNN architectures for image classification because it can lead to better generalization.
* **Adam / AdamW**: The default choice for training Transformer models, NLP, and large generative diffusion networks due to training stability.

## 11. Common Mistakes

* **Adam with weight decay**: Adding weight decay directly to the loss when using Adam. This scales the weight decay factor by the historical gradient variance, which is incorrect. Use AdamW instead.
* **Too high learning rate**: Setting $\eta$ too high. This causes the parameter updates to overshoot the minimum, leading to loss divergence.
* **No bias correction in Adam**: Omitting $\beta_1^t$ and $\beta_2^t$ scaling. This causes updates to be extremely small during the first few training steps.

## 12. Exercises

### Conceptual Questions
1. Why does AdaGrad learning rate decay to zero over time? How does RMSprop resolve this?
2. Explain the difference between Adam and AdamW regarding weight decay.

### Mathematical Exercises
1. Derive the bias correction terms for $m_t$ and $v_t$ in Adam under the assumption that $m_0 = 0$ and $v_0 = 0$.
2. Write down the parameter update equations for Nesterov momentum.

### Programming Exercises
1. Implement the `RMSprop` class in Gradience. Include parameters for `lr`, `alpha` (decay rate), and `eps`.

### Debugging Exercises
1. An Adam optimizer is written but the step counter `self.t` is not incremented. Diagnose the issue and fix the code.

### Research Questions
1. How does the LAMB optimizer scale Adam updates for training with very large batch sizes?

## 13. References

* Qian, N. (1999). On the momentum term in gradient descent algorithms. *Neural Networks*, 12(1), 145-151.
* Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. *Journal of Machine Learning Research*, 12, 2121-2159.
* Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.
* Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*.
