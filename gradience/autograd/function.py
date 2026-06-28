from gradience.autograd.context import Context
from gradience.autograd.graph_node import GraphNode

class Function:
    
    @classmethod
    def apply(cls, *inputs, **kwargs):
        from gradience.tensor import Tensor
        
        ctx = Context()
        raw_inputs = tuple(
            tensor.data
            for tensor in inputs
        )
        result = cls.forward(ctx, *raw_inputs, **kwargs)
        requires_grad=cls._requires_grad(inputs)
        
        output = Tensor._from_operation(
            result,
            requires_grad=requires_grad,
            grad_fn=None
        )
        
        node = GraphNode(
            operation=cls,
            parents=inputs,
            context=ctx,
            output=output
        )
        
        output._grad_fn = node
        
        return output
        
    @staticmethod
    def _requires_grad(inputs):
        return any(
            tensor.requires_grad
            for tensor in inputs
        )
        