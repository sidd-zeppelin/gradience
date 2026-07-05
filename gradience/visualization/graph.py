class GraphNode:
    def __init__(self, node_id, node_type, label, metadata=None):
        self.id = node_id
        self.type = node_type
        self.label = label
        self.metadata = metadata or {}


class GraphEdge:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination


class ComputationGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node):
        self.nodes[node.id] = node

    def add_edge(self, edge):
        self.edges.append(edge)
