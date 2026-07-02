from gradience.nn.losses.loss import Loss

class L1Loss(Loss):

    def forward(self, preds, targets):
        diff = preds - targets
        return diff.abs().mean()
