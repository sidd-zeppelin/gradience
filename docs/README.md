# Gradience Deep Learning Curriculum & Roadmap

Gradience is an educational deep learning framework built from first principles. Its goal is to teach deep learning concepts by implementing them from scratch.

This documentation serves as a complete textbook curriculum. It maps theoretical concepts directly to their practical implementation in Python.

## The Learning Roadmap

The curriculum is structured into the following sequential chapters. Each chapter motivates the necessity of its abstractions, derives the underlying mathematics, and guides you through the first-principles implementation.

* **[Chapter 1: Introduction to Deep Learning](chapters/chapter_1_introduction.md)**: Conceptual foundations, biological and artificial neurons, perceptrons, multi-layer perceptrons, and the universal approximation theorem.
* **[Chapter 2: Building the Foundations](chapters/chapter_2_foundations.md)**: Tensors, computational graphs, backpropagation, and reverse-mode automatic differentiation (autograd).
* **[Chapter 3: Neural Networks](chapters/chapter_3_neural_networks.md)**: Linear layers, activation functions (ReLU, Sigmoid, Tanh), loss functions (MSE, BCE, Cross-Entropy), and multi-layer perceptrons (MLPs).
* **[Chapter 4: Optimization](chapters/chapter_4_optimization.md)**: Gradient descent variants, Stochastic Gradient Descent (SGD), momentum, Nesterov momentum, AdaGrad, RMSprop, Adam, AdamW, and learning rate scheduling.
* **[Chapter 5: Training Neural Networks](chapters/chapter_5_training.md)**: Weight initialization (Xavier, Kaiming/He), normalization layers (BatchNorm, LayerNorm), regularization (Dropout), datasets, dataloaders, and the training pipeline.
* **[Chapter 6: Computer Vision Foundations](chapters/chapter_6_computer_vision.md)**: Image representation as tensors, 2D spatial convolution math, padding, stride, and pooling (MaxPool2D, AdaptiveAvgPool2D).
* **[Chapter 7: CNN Architectures](chapters/chapter_7_cnn_architectures.md)**: Reconstructing classic networks, LeNet-5 for MNIST, and original split-device dual-stream AlexNet (channel concatenation via ConcatOp).

## Curriculum Philosophy

1. **Abstractions must be earned**: No layer, operation, or abstraction is introduced without explaining the limitation of prior methods first.
2. **First-principles derivation**: Every algorithm, forward pass, and gradient update is derived algebraically.
3. **From math to code**: Mathematical variables map directly to code arrays, shape dimensions, and broadcasting mechanisms.
4. **Nothing is a black box**: The code is completely self-contained within this repository and has zero external machine learning dependencies.

## Advanced Roadmap Path

Future additions to the roadmap include:
* Phase 8: Image Segmentation (U-Net, DeepLab)
* Phase 9: Attention Mechanisms
* Phase 10: Transformers (Vision Transformers)
* Phase 11: Generative Models (VAEs, GANs)
* Phase 12: Diffusion Models (DDPM, Score Matching, Stable Diffusion)
