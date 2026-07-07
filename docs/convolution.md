# 2D Convolution Support

This document outlines the design, mathematical formulation, and architecture of the 2D Convolution (`Conv2D`) subsystem in the `Gradience` framework.

---

## Architectural Decomposition

To ensure modularity, separation of concerns, and clean testing, the 2D Convolution implementation is divided into three distinct layers:

1.  **Utilities (`gradience/utils/convolution.py`)**:
    Contains standalone mathematical operations (output shape calculation, zero padding, patch extraction) that are decoupled from both PyTorch-style neural network layers and the autograd engine. This allows them to be reused by future components like `MaxPool2D` or `Deconv2D` layers.
2.  **Autograd Operation (`gradience/ops/conv2d.py`)**:
    Implements `Conv2DOp`, a subclass of `Function`. It handles the raw mathematical computations of the forward and backward passes. It contains absolutely no module state or parameter initialization.
3.  **Stateful Layer (`gradience/nn/convolution/conv2d.py`)**:
    Implements `Conv2D`, a subclass of `Module`. It owns the trainable weight and bias parameters, handles their initialization, defines the PyTorch-compatible constructor API, and invokes `Conv2DOp` on the inputs.

---

## Mathematical Formulation

For a single batch item $n$ and output filter $f$, the forward convolution output at location $(r, c)$ is computed as:

$$y[n, f, r, c] = \sum_{c'=0}^{C-1} \sum_{i=0}^{KH-1} \sum_{j=0}^{KW-1} x_{\text{padded}}[n, c', r \cdot S + i, c \cdot S + j] \cdot w[f, c', i, j] + b[f]$$

Where:
*   $x_{\text{padded}}$ is the padded input tensor.
*   $w$ is the filter weights tensor.
*   $b$ is the bias vector.
*   $S$ is the stride.
*   $C$ is the number of input channels.
*   $KH, KW$ are the kernel spatial dimensions.

### Analytical Gradients (Backward Pass)

Given the incoming gradient $\frac{\partial L}{\partial y}$ (of shape `(N, F, H_out, W_out)`), we compute the analytical gradients for inputs, weights, and bias:

1.  **Bias Gradient**:
    $$\frac{\partial L}{\partial b[f]} = \sum_{n=0}^{N-1} \sum_{r=0}^{H_{\text{out}}-1} \sum_{c=0}^{W_{\text{out}}-1} \frac{\partial L}{\partial y[n, f, r, c]}$$

2.  **Weight Gradient**:
    $$\frac{\partial L}{\partial w[f, c', i, j]} = \sum_{n=0}^{N-1} \sum_{r=0}^{H_{\text{out}}-1} \sum_{c=0}^{W_{\text{out}}-1} \frac{\partial L}{\partial y[n, f, r, c]} \cdot x_{\text{padded}}[n, c', r \cdot S + i, c \cdot S + j]$$

3.  **Padded Input Gradient**:
    $$\frac{\partial L}{\partial x_{\text{padded}}[n, c', h_p, w_p]} = \sum_{f=0}^{F-1} \sum_{\substack{r, c \\ \text{s.t. } h_p = r \cdot S + i \\ w_p = c \cdot S + j}} \frac{\partial L}{\partial y[n, f, r, c]} \cdot w[f, c', i, j]$$

The input gradient $\frac{\partial L}{\partial x}$ is then retrieved by cropping out the padded margins from $\frac{\partial L}{\partial x_{\text{padded}}}$:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial x_{\text{padded}}}[:, :, P : -P, P : -P]$$

---

## Tensor Layouts

Following PyTorch conventions, we adopt the standard four-dimensional layouts:

| Variable | Layout Notation | Dimension Description |
| :--- | :--- | :--- |
| **Input** ($x$) | `(N, C, H, W)` | `(Batch Size, Input Channels, Height, Width)` |
| **Weights** ($w$) | `(F, C, KH, KW)` | `(Output Channels, Input Channels, Kernel Height, Kernel Width)` |
| **Bias** ($b$) | `(F,)` | `(Output Channels,)` |
| **Output** ($y$) | `(N, F, H_out, W_out)` | `(Batch Size, Output Channels, Output Height, Output Width)` |

---

## Naïve Implementation Details

To maximize educational clarity, the forward and backward passes use nested loops mapping directly to the mathematical expressions above:
1.  **Forward Pass**: Iterates over batch items, output channels, and output height/width locations. At each step, it crops a spatial patch of size `(C, KH, KW)` from the padded input and performs an element-wise product and sum with the corresponding filter kernel.
2.  **Backward Pass**: Accumulates weight, bias, and input gradients by iterating over batch items, output filters, and output height/width locations, transferring gradients back to the respective weight locations and input patch coordinates.

---

## Future Roadmap

The naïve convolution is designed as a baseline reference. The roadmap for optimization is:

```
  Naïve Conv2D (Loops over patches)
               │
               ▼
      im2col & col2im (Vectorized matrix-multiplication)
               │
               ▼
  Optimized Conv2D (BLAS GEMM, Winograd/FFT)
```

The division between `Conv2D(Module)` and `Conv2DOp(Function)` ensures that when the vectorized `im2col` backend is implemented, we can replace the underlying `Conv2DOp` implementation without altering the user-facing `Conv2D` layer API.
