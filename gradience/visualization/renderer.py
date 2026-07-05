import graphviz


class GraphRenderer:
    def __init__(self, graph):
        self.graph = graph

    def render(self):
        dot = graphviz.Digraph(graph_attr={"rankdir": "LR"})

        for node_id, node in self.graph.nodes.items():
            if node.type == "tensor":
                dot.node(
                    node.id,
                    label=node.label,
                    shape="ellipse",
                )
            elif node.type == "operation":
                dot.node(
                    node.id,
                    label=node.label,
                    shape="box",
                )

        for edge in self.graph.edges:
            dot.edge(edge.source, edge.destination)

        return dot
