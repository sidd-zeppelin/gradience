class Optimizer:

    def __init__(self, parameters):
        self.parameters = list(parameters)

    def step(self):
        raise NotImplementedError(
            "Optimizer subclasses must implement step()."
        )

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()