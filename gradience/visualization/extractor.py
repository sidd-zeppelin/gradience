from gradience.visualization.graph import ComputationGraph, GraphNode, GraphEdge


class GraphExtractor:
    def __init__(self):
        self.graph = ComputationGraph()
        self.visited_tensors = set()
        self.visited_ops = set()
        self.visited_edges = set()

    def extract(self, tensor):
        self._traverse(tensor)
        return self.graph

    def _traverse(self, tensor):
        tensor_id = f"tensor_{id(tensor)}"
        if tensor_id in self.visited_tensors:
            return

        self.visited_tensors.add(tensor_id)

        metadata = {
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
            "requires_grad": tensor.requires_grad,
        }
        if tensor.size <= 4:
            metadata["preview"] = tensor.data.tolist()

        label = f"Tensor\nshape: {tensor.shape}\ndtype: {tensor.dtype}\nrequires_grad: {tensor.requires_grad}"
        if "preview" in metadata:
            label += f"\nval: {metadata['preview']}"

        node = GraphNode(
            node_id=tensor_id,
            node_type="tensor",
            label=label,
            metadata=metadata,
        )
        self.graph.add_node(node)

        if tensor.grad_fn is not None:
            op_id = f"op_{id(tensor.grad_fn)}"

            if op_id not in self.visited_ops:
                self.visited_ops.add(op_id)
                op_name = tensor.grad_fn.operation.__name__

                op_node = GraphNode(
                    node_id=op_id,
                    node_type="operation",
                    label=op_name,
                    metadata={"op_class": op_name},
                )
                self.graph.add_node(op_node)

                for parent in tensor.grad_fn.parents:
                    self._traverse(parent)
                    parent_id = f"tensor_{id(parent)}"
                    edge_key = (parent_id, op_id)
                    if edge_key not in self.visited_edges:
                        self.visited_edges.add(edge_key)
                        self.graph.add_edge(GraphEdge(parent_id, op_id))

            edge_key = (op_id, tensor_id)
            if edge_key not in self.visited_edges:
                self.visited_edges.add(edge_key)
                self.graph.add_edge(GraphEdge(op_id, tensor_id))
