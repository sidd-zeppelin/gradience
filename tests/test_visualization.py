import graphviz
from gradience.tensor import Tensor
from gradience.visualization import ComputationGraph, GraphExtractor, GraphRenderer


def test_single_tensor():
    x = Tensor(2.0, requires_grad=True)
    extractor = GraphExtractor()
    graph = extractor.extract(x)
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 0
    node = list(graph.nodes.values())[0]
    assert node.type == "tensor"
    assert node.metadata["requires_grad"] is True
    assert node.metadata["shape"] == ()
    assert node.metadata["preview"] == 2.0


def test_basic_operation():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    z = x + y
    extractor = GraphExtractor()
    graph = extractor.extract(z)
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 3
    tensors = [n for n in graph.nodes.values() if n.type == "tensor"]
    ops = [n for n in graph.nodes.values() if n.type == "operation"]
    assert len(tensors) == 3
    assert len(ops) == 1
    assert ops[0].label == "AddOp"


def test_duplicate_prevention():
    x = Tensor(2.0, requires_grad=True)
    z = x + x
    extractor = GraphExtractor()
    graph = extractor.extract(z)
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    tensors = [n for n in graph.nodes.values() if n.type == "tensor"]
    ops = [n for n in graph.nodes.values() if n.type == "operation"]
    assert len(tensors) == 2
    assert len(ops) == 1
    assert ops[0].label == "AddOp"


def test_rendering():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    z = x * y + x
    dot = z.visualize()
    assert isinstance(dot, graphviz.Digraph)
    source = dot.source
    assert "ellipse" in source
    assert "box" in source
