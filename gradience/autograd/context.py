from __future__ import annotations

class Context: # stores info from forward pass that will be needed during backward pass
    
    def __init__(self):
        self._saved_tensors = ()    
        
    # tuple because saved tensors should be read only. 
    # We copy the references not the tensors.
        
    def save_for_backward(self, *tensors):
        self._saved_tensors = tensors
        
    @property                   # read only property
    def saved_tensors(self):
        return self._saved_tensors
    
    def __repr__(self):
        return f"Context(saved_tensors={len(self._saved_tensors)})"