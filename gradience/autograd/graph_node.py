from __future__ import annotations

class GraphNode:
    def __init__(
        self,
        operation,
        parents,
        context,
        output,
    ):
        self.operation = operation
        self.parents = parents
        self.context = context
        self.output = output
        