import numpy as np
from gradience.nn.module import Module
from gradience.nn.parameter import Parameter
from gradience.nn import init
from gradience.ops.conv2d import Conv2DOp


class Conv2D(Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        *,
        weight_initializer=init.he_uniform,
        bias_initializer=init.zeros,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = Parameter(
            np.empty((out_channels, in_channels, kernel_size, kernel_size))
        )
        weight_initializer(self.weight)

        if bias:
            self.bias = Parameter(np.empty(out_channels))
            bias_initializer(self.bias)
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return Conv2DOp.apply(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
            )
        else:
            return Conv2DOp.apply(
                x,
                self.weight,
                stride=self.stride,
                padding=self.padding,
            )

    def __repr__(self):
        return (
            f"Conv2D("
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"bias={self.bias is not None}"
            f")"
        )
