from gradience.nn.module import Module
from gradience.ops.maxpool2d import MaxPool2DOp
from gradience.ops.adaptive_avgpool2d import AdaptiveAvgPool2DOp


class MaxPool2D(Module):

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        return MaxPool2DOp.apply(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

    def __repr__(self):
        return (
            f"MaxPool2D("
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}"
            f")"
        )


class AdaptiveAvgPool2D(Module):

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return AdaptiveAvgPool2DOp.apply(
            x,
            output_size=self.output_size,
        )

    def __repr__(self):
        return f"AdaptiveAvgPool2D(output_size={self.output_size})"
