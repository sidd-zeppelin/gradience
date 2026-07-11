from gradience.nn.module import Module
from gradience.nn.convolution.conv2d import Conv2D
from gradience.nn.layers.pooling import MaxPool2D
from gradience.nn.layers.linear import Linear
from gradience.nn.activations.tanh import Tanh

class LeNet5(Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = Conv2D(in_channels, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = Conv2D(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool = MaxPool2D(kernel_size=2, stride=2)
        
        self.fc3 = Linear(16 * 5 * 5, 120)
        self.fc4 = Linear(120, 84)
        self.fc5 = Linear(84, num_classes)
        self.tanh = Tanh()

    def forward(self, x):
        x = self.pool(self.tanh(self.conv1(x)))
        x = self.pool(self.tanh(self.conv2(x)))
        x = x.flatten(1)
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))
        x = self.fc5(x)
        return x
