from gradience.nn.module import Module

class Activation(Module):
    operation = None    

    def forward(self, x):
        if self.operation is None:
            raise NotImplementedError(
            "Activation subclasses must define an operation"
            )

        return self.operation.apply(x)

    def __repr__(self):
        return f"{self.__class__.__name__}()"