from gradience.nn.losses.loss import Loss

class CrossEntropyLoss(Loss):
    def forward(self, preds, targets):
        from gradience.ops.cross_entropy import CrossEntropy
        return CrossEntropy.apply(preds, targets)
