import numpy as np
from gradience.autograd.function import Function

class CrossEntropy(Function):
    
    @staticmethod
    def forward(ctx, preds, targets):
        shifted_logits = preds - np.max(preds, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        batch_size = preds.shape[0]
        
        if targets.ndim == 1 or (targets.ndim == 2 and targets.shape[1] == 1):
            targets = targets.reshape(-1).astype(int)
            log_preds = np.log(probs[np.arange(batch_size), targets] + 1e-15)
            ctx.save_for_backward(probs, targets, np.array(True))
        else:
            log_preds = np.log(np.sum(probs * targets, axis=-1) + 1e-15)
            ctx.save_for_backward(probs, targets, np.array(False))             
        loss = -np.mean(log_preds)
        return np.array(loss)
        
    @staticmethod
    def backward(ctx, grad_output):
        probs, targets, is_indices = ctx.saved_tensors
        batch_size = probs.shape[0]
        grad_preds = probs.copy()
        
        if is_indices:
            grad_preds[np.arange(batch_size), targets] -= 1.0
        else:
            grad_preds -= targets
            
        grad_preds = grad_preds / batch_size
        return grad_output * grad_preds, None
