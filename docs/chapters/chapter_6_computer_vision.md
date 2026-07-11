# Chapter 6: Computer Vision Foundations

This chapter introduces the fundamental operations of convolutional neural networks. We define image representations. We derive 2D spatial convolutions. We analyze padding, stride, and spatial pooling. Finally, we formulate receptive fields.

## 1. Historical Motivation

Fully connected Multi-Layer Perceptrons perform poorly on image datasets. First, they do not preserve spatial structure. An image is represented as a flat 1D vector. This destroys the two-dimensional relationships between neighboring pixels. Second, fully connected layers suffer from parameter explosion. An input image of size $256 \times 256 \times 3$ features 196,608 dimensions. A single linear layer with 1000 hidden units requires nearly 200 million parameters. This leads to overfitting and makes optimization impossible.

Third, fully connected layers are not translation invariant. If an object shifts by a few pixels, the input vector changes completely. The network must learn to recognize the object at every possible pixel coordinate.

Convolutional layers solve these issues. They use weight sharing and local receptive fields. A small filter kernel slides across the image, computing local patterns. This preserves spatial topology. It also drastically reduces the parameter count and provides translation invariance.

## 2. Intuition

Imagine you are looking at a painting through a small cardboard tube. You can only see a small $3 \times 3$ grid of the canvas at one time. As you slide the tube across the canvas from left to right, you look for specific features. For example, you look for vertical edges or color transitions.

This sliding window is a convolution filter. The cardboard tube size is the kernel size. The amount you slide the tube at each step is the stride. 

If you want to inspect the canvas borders, the cardboard tube might fall off the edge. To prevent this, you can attach a blank paper border around the painting. This is padding.

After detecting features at every canvas location, you want to summarize the results. Instead of keeping the exact coordinates of every edge, you record the maximum edge intensity in each small region. This is max pooling. It reduces the spatial resolution of the feature map while keeping the most prominent features.

## 3. Mathematical Foundations

Let $x \in \mathbb{R}^{B \times C_{\text{in}} \times H \times W}$ be the input tensor, where $B$ is the batch size, $C_{\text{in}}$ is the number of input channels, and $H, W$ are the spatial height and width. Let $w \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times KH \times KW}$ be the weight kernel, where $C_{\text{out}}$ is the number of output filters, and $KH, KW$ are the kernel dimensions. Let $b \in \mathbb{R}^{C_{\text{out}}}$ be the bias vector.

### 1. Zero Padding

We add $P$ rows and columns of zeros to the borders of the spatial dimensions. For padding $P$, the padded height $H_{\text{pad}}$ and width $W_{\text{pad}}$ are:

$$H_{\text{pad}} = H + 2P, \quad W_{\text{pad}} = W + 2P$$

During backpropagation, we crop the incoming gradient to the original shape to discard the padding borders:

$$\frac{\partial L}{\partial x} = \text{crop}\left(\frac{\partial L}{\partial x_{\text{pad}}}, \text{from } P \text{ to } H+P\right)$$

### 2. Forward Convolution

For a single batch index $n$, filter $f$, and output spatial location $(r, c)$, the forward pass is:

$$y[n, f, r, c] = \sum_{c'=0}^{C_{\text{in}}-1} \sum_{i=0}^{KH-1} \sum_{j=0}^{KW-1} x_{\text{padded}}[n, c', r \cdot S + i, c \cdot S + j] \cdot w[f, c', i, j] + b[f]$$

where $S$ is the stride factor.

The output spatial dimensions $H_{\text{out}}$ and $W_{\text{out}}$ are:

$$H_{\text{out}} = \left\lfloor \frac{H + 2P - KH}{S} \right\rfloor + 1$$
$$W_{\text{out}} = \left\lfloor \frac{W + 2P - KW}{S} \right\rfloor + 1$$

### 3. Backward Convolution Gradients

During the backward pass, we receive the incoming gradient $\frac{\partial L}{\partial y} \in \mathbb{R}^{B \times C_{\text{out}} \times H_{\text{out}} \times W_{\text{out}}}$. 

We compute the parameter gradients by accumulating the sliding patch multiplications:

$$\frac{\partial L}{\partial w[f, c', i, j]} = \sum_{n=0}^{B-1} \sum_{r=0}^{H_{\text{out}}-1} \sum_{c=0}^{W_{\text{out}}-1} \frac{\partial L}{\partial y[n, f, r, c]} \cdot x_{\text{padded}}[n, c', r \cdot S + i, c \cdot S + j]$$
$$\frac{\partial L}{\partial b[f]} = \sum_{n=0}^{B-1} \sum_{r=0}^{H_{\text{out}}-1} \sum_{c=0}^{W_{\text{out}}-1} \frac{\partial L}{\partial y[n, f, r, c]}$$

We compute the input gradient by sliding the weight kernels over the output gradient maps:

$$\frac{\partial L}{\partial x_{\text{padded}}[n, c', r', c']} = \sum_{f=0}^{C_{\text{out}}-1} \sum_{i=0}^{KH-1} \sum_{j=0}^{KW-1} \frac{\partial L}{\partial y[n, f, r, c]} \cdot w[f, c', i, j]$$

where $r' = r \cdot S + i$ and $c' = c \cdot S + j$.

### 4. Pooling Layers

* **MaxPool2D**: Extract the maximum value in each $KH \times KW$ spatial patch.
  During backpropagation, we route the incoming gradient entirely to the spatial coordinate that contained the maximum value during the forward pass.
* **AdaptiveAvgPool2D**: Target an output shape $OH \times OW$. The stride and kernel sizes are automatically calculated to divide the input spatial dimensions evenly.
  During backpropagation, we distribute the incoming gradient evenly across the pixels of each local pool region.

## 4. Mathematical Intuition

Let us examine the stride and padding dimension formula:

$$H_{\text{out}} = \frac{H + 2P - KH}{S} + 1$$

The numerator $H + 2P - KH$ represents the total spatial range over which the kernel can slide. 

Dividing by $S$ calculates the number of step updates. 

Adding $1$ includes the initial position of the kernel before any steps are taken. If the division results in a fraction, we apply the floor function. This discards any boundary region that is too small for a complete kernel step.

## 5. From Mathematics to Code

We implement the convolution operations using nested spatial loops in a custom `Function` subclass.

Let us map the zero padding operation.
Mathematics:
Add $P$ zeros to dimensions $H$ (axis 2) and $W$ (axis 3) of $X$.

Code implementation:
```python
# During forward pass:
X_padded = np.pad(X, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')

# During backward pass (cropping the gradient):
grad_x = grad_output_padded[:, :, P:-P, P:-P]
```

## 6. Gradience Implementation

Here is the forward pass implementation of `Conv2DOp` from the framework. It handles padded array construction and slides the filter kernels.

```python
class Conv2DOp(Function):
    @staticmethod
    def forward(ctx, x, w, b, padding=0, stride=1):
        ctx.padding = padding
        ctx.stride = stride
        ctx.save_for_backward(x, w, b)
        
        B, C_in, H, W = x.data.shape
        C_out, _, KH, KW = w.data.shape
        
        H_out = (H + 2 * padding - KH) // stride + 1
        W_out = (W + 2 * padding - KW) // stride + 1
        
        x_pad = np.pad(x.data, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
        out = np.zeros((B, C_out, H_out, W_out))
        
        for n in range(B):
            for f in range(C_out):
                for r in range(H_out):
                    for c in range(W_out):
                        r_start = r * stride
                        c_start = c * stride
                        patch = x_pad[n, :, r_start:r_start+KH, c_start:c_start+KW]
                        out[n, f, r, c] = np.sum(patch * w.data[f]) + (b.data[f] if b is not None else 0.0)
                        
        return Tensor(out)
```
The forward method loops over batch index $n$, filter index $f$, and output spatial coordinates $(r, c)$. It extracts the local input patch and evaluates the element-wise multiplication and sum.

## 7. Complexity Analysis

Let $B$ be the batch size, $C_{\text{in}}, C_{\text{out}}$ be the channels, $H, W$ be the input dimensions, and $KH, KW$ be the kernel dimensions.

* **Time Complexity**:
  * Forward Pass: $O(B \cdot C_{\text{out}} \cdot H_{\text{out}} \cdot W_{\text{out}} \cdot C_{\text{in}} \cdot KH \cdot KW)$ due to the nested spatial loops.
  * Backward Pass: $O(B \cdot C_{\text{out}} \cdot H_{\text{out}} \cdot W_{\text{out}} \cdot C_{\text{in}} \cdot KH \cdot KW)$ to compute the gradients of the weights and inputs.
* **Space Complexity**:
  * Parameter storage: $O(C_{\text{out}} \cdot C_{\text{in}} \cdot KH \cdot KW + C_{\text{out}})$ for weights and biases.
  * Cache size: $O(B \cdot C_{\text{in}} \cdot H \cdot W)$ to store inputs for the backward pass.

## 8. Visualizations

Here is a diagram showing the spatial convolution operation with stride 1 and padding 0:

```
Input Image (3x3)      Kernel (2x2)       Output (2x2)
   +---+---+---+          +---+---+          +---+---+
   | a | b | c |    *     | w | x |    =     | 1 | 2 |
   +---+---+---+          +---+---+          +---+---+
   | d | e | f |          | y | z |          | 3 | 4 |
   +---+---+---+          +---+---+          +---+---+
   | g | h | i |
   +---+---+---+

   Output 1 = a*w + b*x + d*y + e*z
```

## 9. Comparisons

| Layer | Weight Parameters | Spatial Invariance | Primary Feature |
| :--- | :--- | :--- | :--- |
| Linear | $O(\text{Input} \cdot \text{Output})$ | None | Learns global relationships |
| Conv2D | $O(\text{Kernel\_size} \cdot C_{\text{in}} \cdot C_{\text{out}})$ | Yes | Extracts local spatial structures |

## 10. Practical Applications

Convolutional layers are used in:
* **Feature Extraction**: Extracting edges, textures, and shapes in computer vision networks.
* **Spatial Reduction**: Downsampling representations using pooling layers.
* **Image Synthesis**: Generative adversarial networks and diffusion layers.

## 11. Common Mistakes

* **Incorrect Output Shapes**: Setting kernel, padding, and stride combinations that result in fractional output spatial dimensions.
* **Channel Mismatch**: Specifying an input channel dimension in the convolutional weights that does not match the channel dimension of the incoming feature map.
* **Neglecting Padding Gradients**: Forgetting to crop the padding borders from the input gradient during backpropagation. This creates shape mismatches in the computational graph.

## 12. Exercises

### Conceptual Questions
1. Why does weight sharing in convolutional layers reduce the risk of overfitting compared to fully connected layers?
2. Explain the difference between MaxPool2D and AveragePool2D during backpropagation.

### Mathematical Exercises
1. Calculate the output spatial shape for an input image of size $224 \times 224$ passing through a Conv2D layer with kernel size 11, stride 4, and padding 2.
2. Derive the backward gradient updates for the convolutional weights when stride $S > 1$.

### Programming Exercises
1. Implement the backward method for the `Conv2DOp` class in NumPy. Verify the gradients using the numerical gradcheck utility.

### Debugging Exercises
1. A convolutional layer forward pass throws an index out of bounds error. The input shape is $32 \times 32$, kernel size is 5, padding is 0, and stride is 2. Analyze the shape computation.

### Research Questions
1. How does dilated convolution expand the receptive field without increasing the parameter count?

## 13. References

* LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
* Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. *arXiv preprint arXiv:1603.07285*.
