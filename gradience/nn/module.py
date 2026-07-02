from gradience.nn.parameter import Parameter

class Module:
    def __init__(self):
        self.training = True
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "You must implement the forward method in your custom layer!"
        )
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    def parameters(self):
        params = []
        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params
        
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    def train(self):
        self.training = True
        
        for value in self.__dict__.values():
            if isinstance(value, Module):
                value.train()

    def eval(self):
        self.training = False
        
        for value in self.__dict__.values():
            if isinstance(value, Module):
                value.eval()