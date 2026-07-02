# Gradience Roadmap

This document outlines the planned progression of the Gradience framework.

---

## Phase 5: Advanced Architectures (v0.3.0)
* **Convolutional Layers**: Implement `Conv2d` and `MaxPool2d` for processing spatial grid data (like images). This will utilize the `im2col` trick to map local receptive fields into columns for highly optimized matrix multiplications.
* **Recurrent Networks**: Implement `RNN`, `LSTM`, and `GRU` modules for sequential and time-series data.
* **Embedding Layers**: Support token-to-vector lookups for Natural Language Processing tasks.

## Phase 6: Training Utilities (v0.4.0)
* **Model Serialization**: Implement `save()` and `load()` functions to persist model parameters (`weight` and `bias`) to disk using NumPy's serialization format.
* **Data Pipelines**: Create `Dataset` and `DataLoader` abstractions to handle batching, shuffling, and streaming data during training epochs.
* **Learning Rate Schedulers**: Implement learning rate decay strategies (e.g., `StepLR`, `CosineAnnealingLR`) to modulate learning rates dynamically.

## Phase 7: Optimization & Extensions (v0.5.0)
* **In-place Operations**: Implement memory-efficient operations like `relu_` (in-place ReLU) to avoid allocating new memory buffers.
* **Memory Management**: Add eager graph cleanup and reference releasing post-backward pass.
* **Functional API**: Provide a functional namespace (`gradience.nn.functional`) containing stateless versions of all activations, losses, and normalization layers.
