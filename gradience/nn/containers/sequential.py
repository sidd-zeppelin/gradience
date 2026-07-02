from gradience.nn.module import Module
from gradience.nn.parameter import Parameter


class Sequential(Module):

    def __init__(self, *_modules):
        super().__init__()

        self._modules = list(_modules)

    def forward(self, x):
        for module in self._modules:
            x = module(x)
        return x

    def parameters(self):
        parameters = []

        for value in self.__dict__.values():

            if isinstance(value, Parameter):
                parameters.append(value)

            elif isinstance(value, Module):
                parameters.extend(value.parameters())

            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        parameters.extend(item.parameters())

        return parameters
    def train(self):
        super().train()
        for module in self._modules:
            module.train()

    def eval(self):
        super().eval()
        for module in self._modules:
            module.eval()

    def __len__(self):
        return len(self._modules)

    def __getitem__(self, index):
        return self._modules[index]

    def __repr__(self):
        body = ",\n  ".join(
            repr(module)
            for module in self._modules
        )

        return f"Sequential(\n  {body}\n)"