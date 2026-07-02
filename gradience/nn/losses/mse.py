from gradience.nn.module import Module

class MSELoss(Module):

    def forward(
        self,
        prediction,
        target,
    ):
        return (
            (prediction - target) ** 2
        ).mean()

    def __repr__(self):
        return "MSELoss()"