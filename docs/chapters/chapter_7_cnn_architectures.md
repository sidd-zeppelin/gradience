# Chapter 7: CNN Architectures

This chapter analyzes two milestone convolutional neural network architectures. We study LeNet-5. We detail the original dual-stream parallel AlexNet. Finally, we formulate the autograd concatenation operation.

## 1. Historical Motivation

Early neural network designs were limited to simple pattern recognition on small images. LeCun et al. (1998) introduced LeNet-5, which successfully recognized handwritten digits on checks. LeNet-5 established the core template of modern CNNs: stacking convolutional layers, activation functions, pooling layers, and fully connected heads.

For over a decade, computer vision still relied on hand-crafted features (like SIFT or HOG) because training deeper networks on larger datasets was computationally impossible. In 2012, Krizhevsky et al. introduced AlexNet, which won the ImageNet competition by a massive margin. AlexNet was deeper, utilized ReLU activations to speed up training, and employed dropout to prevent overfitting.

However, the hardware available in 2012 (the NVIDIA GTX 580 GPU) had only 3 gigabytes of VRAM. This memory was too small to store the activations and parameters of AlexNet. To solve this hardware limitation, the authors split the model channels across two parallel streams running on two separate GPUs. The streams only communicated at specific layers using cross-device concatenation.

## 2. Intuition

Imagine you want to assemble a large, complex puzzle, but your workspace table is too small to fit the entire puzzle. 

To solve this, you split the puzzle into two halves. You invite a friend, and each of you works on one half of the puzzle on two separate tables. This is parallel streaming.

Most of the time, you and your friend work independently on your respective tables. However, some pieces near the boundary require looking at the other half of the puzzle. At specific stages, you and your friend meet at a shared table to align and merge your pieces. After aligning them, you split the work again and return to your tables.

In the original AlexNet, Stream 1 and Stream 2 extract different sets of features. At Layer 3 and Layer 6, the feature maps from both streams are concatenated. This allows the filters to learn cross-stream correlations.

## 3. Mathematical Foundations

Let $x_1 \in \mathbb{R}^{B \times C_1 \times H \times W}$ and $x_2 \in \mathbb{R}^{B \times C_2 \times H \times W}$ be the feature maps from Stream 1 and Stream 2.

### 1. Concatenation Operation (ConcatOp)

The forward pass concatenates the inputs along the channel dimension (axis 1):

$$y = \text{concat}(x_1, x_2) \in \mathbb{R}^{B \times (C_1 + C_2) \times H \times W}$$

During backpropagation, we receive the incoming gradient $\frac{\partial L}{\partial y} \in \mathbb{R}^{B \times (C_1 + C_2) \times H \times W}$. The backward pass slices the gradient along axis 1 to distribute the gradients back to the respective streams:

$$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial y}[:, 0:C_1, :, :]$$
$$\frac{\partial L}{\partial x_2} = \frac{\partial L}{\partial y}[:, C_1:(C_1 + C_2), :, :]$$

### 2. LeNet-5 Architecture

LeNet-5 takes an input of shape $1 \times 32 \times 32$. The sequential structure is:
1. **Conv1**: 6 filters of size $5 \times 5$, stride 1. Output: $6 \times 28 \times 28$.
2. **MaxPool1**: kernel $2 \times 2$, stride 2. Output: $6 \times 14 \times 14$.
3. **Conv2**: 16 filters of size $5 \times 5$, stride 1. Output: $16 \times 10 \times 10$.
4. **MaxPool2**: kernel $2 \times 2$, stride 2. Output: $16 \times 5 \times 5$.
5. **FC3**: Fully connected to 120 units.
6. **FC4**: Fully connected to 84 units.
7. **FC5**: Output layer with 10 class logits.

### 3. Original Dual-Stream AlexNet Architecture

The original AlexNet splits channels into two parallel streams:
* **Input**: $3 \times 224 \times 224$ images.
* **Conv1**: 96 filters of size $11 \times 11$, stride 4, padding 2.
  * Stream 1 Conv1: 48 filters. Output: $48 \times 55 \times 55$.
  * Stream 2 Conv1: 48 filters. Output: $48 \times 55 \times 55$.
  * Both streams apply MaxPool ($3 \times 3$, stride 2). Output: $48 \times 27 \times 27$.
* **Conv2**: 256 filters of size $5 \times 5$, padding 2.
  * Stream 1 Conv2: 128 filters (takes Stream 1 output). Output: $128 \times 27 \times 27$.
  * Stream 2 Conv2: 128 filters (takes Stream 2 output). Output: $128 \times 27 \times 27$.
  * Both apply MaxPool ($3 \times 3$, stride 2). Output: $128 \times 13 \times 13$.
* **Cross-Stream Concatenation (Layer 3 input)**:
  * Concatenate Stream 1 and Stream 2 outputs to form $256 \times 13 \times 13$.
* **Conv3**: 384 filters of size $3 \times 3$, padding 1.
  * Stream 1 Conv3: 192 filters (takes concatenated output). Output: $192 \times 13 \times 13$.
  * Stream 2 Conv3: 192 filters (takes concatenated output). Output: $192 \times 13 \times 13$.
* **Conv4**: 384 filters of size $3 \times 3$, padding 1.
  * Stream 1 Conv4: 192 filters (takes Stream 1 Conv3). Output: $192 \times 13 \times 13$.
  * Stream 2 Conv4: 192 filters (takes Stream 2 Conv3). Output: $192 \times 13 \times 13$.
* **Conv5**: 256 filters of size $3 \times 3$, padding 1.
  * Stream 1 Conv5: 128 filters. Output: $128 \times 13 \times 13$.
  * Stream 2 Conv5: 128 filters. Output: $128 \times 13 \times 13$.
  * Both apply MaxPool ($3 \times 3$, stride 2). Output: $128 \times 6 \times 6$.
* **Cross-Stream Concatenation (FC6 input)**:
  * Concatenate Stream 1 and Stream 2 outputs to form $256 \times 6 \times 6 = 9216$ units.
* **Classifier**:
  * **FC6**: 4096 units (takes concatenated output).
  * **FC7**: 4096 units.
  * **FC8**: Output layer with 10 logits.

## 4. Mathematical Intuition

Let us examine the gradient slicing equation:

$$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial y}[:, 0:C_1, :, :]$$

During concatenation, Stream 1 and Stream 2 features are stacked. No mathematical interactions occur between the channels themselves. 

Therefore, during the backward pass, the gradient flowing through the concatenated channels simply splits. The first $C_1$ channels of the gradient belong to Stream 1. The remaining channels belong to Stream 2. This represents a zero-parameter routing block.

## 5. From Mathematics to Code

We implement the concatenation operation as an autograd `Function` subclass in Gradience.

Let us map the concatenation forward and backward passes.
Forward: Concatenate list of inputs along axis 1.
Backward: Slice incoming gradient along axis 1 using the input shapes.

Code implementation:
```python
# Forward:
out = np.concatenate([x.data for x in inputs], axis=axis)

# Backward (slicing the gradient):
grads = []
start = 0
for shape in input_shapes:
    c = shape[axis]
    end = start + c
    grad_slice = grad_output[:, start:end, :, :]
    grads.append(grad_slice)
    start = end
```

## 6. Gradience Implementation

Here is the implementation of `ConcatOp` from the framework:

```python
from gradience.tensor import Tensor
from gradience.ops.op import Function
import numpy as np

class ConcatOp(Function):
    @staticmethod
    def forward(ctx, *tensors, axis=1):
        ctx.axis = axis
        ctx.input_shapes = [t.data.shape for t in tensors]
        ctx.parents = list(tensors)
        
        arrs = [t.data for t in tensors]
        out = np.concatenate(arrs, axis=axis)
        return Tensor(out)

    @staticmethod
    def backward(ctx, grad_output):
        axis = ctx.axis
        shapes = ctx.input_shapes
        
        grads = []
        start = 0
        for shape in shapes:
            c = shape[axis]
            end = start + c
            
            # Slice along the designated axis
            slices = [slice(None)] * len(shape)
            slices[axis] = slice(start, end)
            
            grad_slice = grad_output[tuple(slices)]
            grads.append(grad_slice)
            start = end
            
        return tuple(grads)
```
The constructor saves the input shapes and the designated concatenation axis in the context. The backward method creates dynamic slice tuples using Python's `slice` objects. This allows slicing along any specified axis.

Here is the implementation of the `LeNet5` model class:

```python
from gradience.nn.module import Module
from gradience.nn.convolution.conv2d import Conv2D
from gradience.nn.layers.pooling import MaxPool2D
from gradience.nn.layers.linear import Linear
from gradience.nn.activations.tanh import Tanh

class LeNet5(Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = Conv2D(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = Conv2D(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool = MaxPool2D(kernel_size=2, stride=2)
        
        self.fc3 = Linear(16 * 5 * 5, 120)
        self.fc4 = Linear(120, 84)
        self.fc5 = Linear(84, num_classes)
        self.tanh = Tanh()

    def forward(self, x):
        x = self.pool(self.tanh(self.conv1(x)))
        x = self.pool(self.tanh(self.conv2(x)))
        x = x.flatten(1)
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))
        x = self.fc5(x)
        return x
```
The forward pass uses the Tanh activation function after the first two convolutions. It flattens the pooled activations before projecting them through fully connected layers.

## 7. Complexity Analysis

Let $B$ be the batch size, $C_1, C_2$ be the channels, and $H, W$ be the spatial dimensions.

* **Time Complexity**:
  * Forward Pass: $O(B \cdot (C_1 + C_2) \cdot H \cdot W)$ due to array allocation and copy.
  * Backward Pass: $O(B \cdot (C_1 + C_2) \cdot H \cdot W)$ to slice the gradient array.
* **Space Complexity**:
  * Output tensor allocation: $O(B \cdot (C_1 + C_2) \cdot H \cdot W)$.
  * Backward state: $O(1)$ since only shapes and axis are cached.

## 8. Visualizations

Here is a diagram showing the original dual-stream AlexNet parallel routing:

```
                  Image Input (3x224x224)
                 /                       \
                v                         v
           Stream 1 Conv1            Stream 2 Conv1
                |                         |
                v                         v
           Stream 1 Conv2            Stream 2 Conv2
                \                         /
                 v                       v
               [ Cross-Stream Concatenation ] (axis 1)
                /                         \
               v                           v
           Stream 1 Conv3            Stream 2 Conv3
                |                         |
                v                         v
           Stream 1 Conv4            Stream 2 Conv4
                |                         |
                v                         v
           Stream 1 Conv5            Stream 2 Conv5
                \                         /
                 v                       v
               [ Cross-Stream Concatenation ] (axis 1)
                            |
                            v
                      Classifier (FC)
```

## 9. Comparisons

| Feature | LeNet-5 | Original AlexNet |
| :--- | :--- | :--- |
| Parallel Streams | No | Yes (2 streams) |
| Activations | Sigmoid/Tanh | ReLU |
| Regularization | None | Dropout |
| Training Hardware | CPU | Multi-GPU split |

## 10. Practical Applications

Concatenation is a key tensor routing operation used in:
* **U-Net**: Skip connections concatenate encoder feature maps with decoder feature maps.
* **DenseNet**: Dense blocks concatenate feature maps from all previous layers.
* **Inception Blocks**: Concatenating outputs of parallel kernels of different sizes.

## 11. Common Mistakes

* **Dimension Mismatch**: Concatenating tensors that have different spatial heights or widths. Concatenation along axis 1 requires all other dimensions to match exactly.
* **Incorrect Slice Ordering**: Reversing the order of sliced gradients during backpropagation. The gradients must be returned in the exact order of the forward inputs.
* **Retaining unnecessary copies**: Creating copies of arrays during concatenation, which increases VRAM usage.

## 12. Exercises

### Conceptual Questions
1. Why did the GTX 580 memory limits necessitate a dual-stream architecture in 2012?
2. Explain how a skip connection in a network uses concatenation to preserve fine-grained spatial features.

### Mathematical Exercises
1. Let $x_1$ be of shape $(2, 3, 4, 4)$ and $x_2$ be of shape $(2, 5, 4, 4)$. Write down the output shape of concatenating them along axis 1.
2. Formulate the backward pass derivative of the concatenation operation if the inputs are concatenated along the batch dimension (axis 0).

### Programming Exercises
1. Write a custom network class `OriginalAlexNet` using Gradience. Ensure weights are synchronized memory-for-memory.

### Debugging Exercises
1. A concatenation backward pass throws a shape mismatch error during parameter updates. The inputs are list of 2 tensors, but the backward method returned a single tensor. Fix the bug.

### Research Questions
1. How does the split-device training of AlexNet in 2012 compare with modern tensor-parallelism techniques?

## 13. References

* LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
* Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems*, 25, 1097-1105.
