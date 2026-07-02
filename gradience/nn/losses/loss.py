from gradience.nn.module import Module


class Loss(Module):

    operation = None

    def forward(self, prediction, target):
        if self.operation is None:
            raise NotImplementedError(
                "Loss subclasses must define an operation."
            )

        return self.operation.apply(
            prediction,
            target,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}()"