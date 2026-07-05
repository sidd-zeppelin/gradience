from gradience.visualization.graph import ComputationGraph, GraphNode, GraphEdge
from gradience.visualization.extractor import GraphExtractor
from gradience.visualization.renderer import GraphRenderer


def visualize(tensor):
    extractor = GraphExtractor()
    graph = extractor.extract(tensor)
    renderer = GraphRenderer(graph)
    return renderer.render()


__all__ = [
    "ComputationGraph",
    "GraphNode",
    "GraphEdge",
    "GraphExtractor",
    "GraphRenderer",
    "visualize",
]
