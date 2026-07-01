import numpy as np

class AutogradEngine:

    @staticmethod
    def _topological_sort(tensor):
        visited = set()
        order = []
        
        def dfs(tensor):
            
            if tensor in visited:
                return

            visited.add(tensor)
            
            if tensor.grad_fn is not None:
                for parent in tensor.grad_fn.parents:
                    dfs(parent)
            
            order.append(tensor)
        dfs(tensor)
        
        return order
    
    @staticmethod
    def backward(tensor):
        order = AutogradEngine._topological_sort(tensor)
        tensor.grad = np.ones_like(tensor.data)
        
        for tensor in reversed(order):
            
            if tensor.grad_fn is None:
                continue
            
            node = tensor.grad_fn
            
            grads = node.operation.backward(
                node.context,
                tensor.grad
            )

            
            for parent, grad in zip(node.parents, grads):
                if not parent.requires_grad:
                    continue
                if parent.grad is None:
                    parent.grad = grad
                else:
                    parent.grad += grad
                
            
            