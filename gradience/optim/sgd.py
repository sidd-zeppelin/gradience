from gradience.optim.optimizer import Optimizer


class SGD(Optimizer):

    def __init__(
        self,
        parameters,
        lr=0.01,
    ):
        super().__init__(parameters)

        self.lr = lr

    def step(self):

        for parameter in self.parameters:

            if parameter.grad is None:
                continue

            parameter._data -= (
                self.lr * parameter.grad
            )