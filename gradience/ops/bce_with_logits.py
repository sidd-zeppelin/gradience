import numpy as np
from gradience.autograd.function import Function

class BCEWithLogits(Function):
    
    @staticmethod
    def forward(ctx, preds, targets):
        
        max_val = np.clip(preds, 0, None)
        loss = max_val - preds * targets + np.log(1 + np.exp(-np.abs(preds)))
        mean_loss = np.mean(loss)
        
        sig_preds = 1 / (1 + np.exp(-preds))
        ctx.save_for_backward(sig_preds, targets)
        
        return np.array(mean_loss)
        
    @staticmethod
    def backward(ctx, grad_output):
        sig_preds, targets = ctx.saved_tensors
        batch_size = np.prod(sig_preds.shape) 
        
        grad_preds = (sig_preds - targets) / batch_size
        return grad_output * grad_preds, None
