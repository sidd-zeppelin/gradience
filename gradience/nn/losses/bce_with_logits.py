from gradience.nn.losses.loss import Loss

class BCEWithLogitsLoss(Loss):

    def forward(self, preds, targets):
        from gradience.ops.bce_with_logits import BCEWithLogits
        return BCEWithLogits.apply(preds, targets)
