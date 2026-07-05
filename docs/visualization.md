# Computational Graph Visualization

This document details the architecture, design choices, and algorithms powering the computational graph visualization subsystem of `Gradience`.

---

## Architecture Overview

The visualization subsystem is designed as an independent module (`gradience.visualization`) completely decoupled from both the autograd execution engine (`gradience.autograd`) and third-party rendering backends (like Graphviz).

This separation of concerns ensures that the core framework code remains clean, and the visualization tools can be extended or replaced without modifying autograd logic.

The design relies on three core components:

```
  Tensor (Autograd Graph)
          │
          ▼
   [ GraphExtractor ]
          │ (Traverses & extracts generic models)
          ▼
  ComputationGraph (Nodes & Edges)
          │
          ▼
   [ GraphRenderer ]
          │ (Constructs visual format)
          ▼
    graphviz.Digraph
```

---

## Core Components

### 1. ComputationGraph (`gradience/visualization/graph.py`)

A pure Python, generic graph model holding elements independent of any layout engine.
* **`GraphNode`**: Represents a node with a unique ID, type (`"tensor"` or `"operation"`), label text, and dictionary of metadata.
* **`GraphEdge`**: Represents a directed edge pointing from a source node ID to a destination node ID.
* **`ComputationGraph`**: An object containing dictionary of nodes and list of edges. It provides utility methods to add nodes and edges.

### 2. GraphExtractor (`gradience/visualization/extractor.py`)

Handles the discovery and translation of `Gradience` models into a generic `ComputationGraph`.
* It recurses backward from a root `Tensor` (such as a loss value).
* It traverses the computation graph by visiting `tensor.grad_fn` and then checking `parents` of the operation node.
* It uses sets of visited nodes and visited edge tuples to avoid duplicate graph elements, ensuring that even if a tensor is reused across multiple operations, it appears as a single node with correct branches.

### 3. GraphRenderer (`gradience/visualization/renderer.py`)

Responsible for generating a visual representation from a generic `ComputationGraph`.
* The default implementation uses `graphviz.Digraph` to create a diagram.
* It renders tensor nodes as **ellipses** and operations as **boxes**.
* Layout uses a left-to-right (`LR`) direction, mapping data flowing from parents to operations to results.

---

## Design Decisions

### Separating Visualization from Autograd
* **Safety**: Merging visualization logic with the core autograd execution engine risks injecting side-effects into the forward and backward passes.
* **Simplicity**: `AutogradEngine` handles mathematical propagation. By keeping it focused strictly on numerical evaluation, we prevent code bloat and preserve its high performance.
* **Alternative Renderers**: The separation makes it trivial to write alternative renderers (e.g. exporting to Mermaid, interactive HTML widgets, or networkx analysis) without changing how the graph structure is extracted.

### Node and Edge Representation
To capture operations clearly, the graph uses explicit bipartite-like connections:
* `Tensor -> Operation -> Tensor`
* Edges represent dependencies and the flow of data.
* Operations are created from `grad_fn` executions. Because `GraphNode` represents a unique step of calculation, naming it by its primitive operation class name (e.g., `AddOp`, `MatMulOp`) matches the code's native design.

### Handling Tensors
* **Leaf Tensors**: Tensors without a `grad_fn` represent parameter or input leaves. They form the start/inputs of the graph.
* **Metadata & Previews**: High-dimensional tensors are summarized (shape, dtype, requires_grad) to avoid visual clutter. Small tensors (size $\le 4$) include a value preview.

---

## Future Roadmap & Extension Points

The decoupled architecture easily accommodates future capabilities:

1. **Gradient Visualizations**:
   * Extend `GraphExtractor` to capture `tensor.grad` and pass it in node metadata.
   * Customize the renderer to color-code gradients based on magnitude (for debugging vanishing/exploding gradients).
2. **Local Jacobians**:
   * Save and visualize the computed local gradients/Jacobians at each node during the backward pass.
3. **Execution Tracing & Execution Order**:
   * Trace and animate the forward/backward flow of execution.
4. **Interactive HTML Renderers**:
   * Replace `GraphRenderer` with a backend that generates SVG, D3.js, or interactive HTML widgets to support tooltips and inspectable parameter properties.
