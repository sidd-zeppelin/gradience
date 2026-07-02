from gradience.nn.parameter import Parameter

class Module:
    def __init__(self):
        pass
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError("You must implement the forward method in your custom layer!")
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    def parameters(self):
        params = []
        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params
        
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()
