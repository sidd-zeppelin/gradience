from gradience.nn.module import Module
from gradience.nn.containers.sequential import Sequential
from gradience.nn.convolution.conv2d import Conv2D
from gradience.nn.layers.linear import Linear
from gradience.nn.layers.dropout import Dropout
from gradience.nn.layers.pooling import MaxPool2D, AdaptiveAvgPool2D
from gradience.nn.activations.relu import ReLU
from gradience.tensor import concat


class AlexNet(Module):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = Sequential(
            Conv2D(3, 64, kernel_size=11, stride=4, padding=2),
            ReLU(),
            MaxPool2D(kernel_size=3, stride=2),
            Conv2D(64, 192, kernel_size=5, padding=2),
            ReLU(),
            MaxPool2D(kernel_size=3, stride=2),
            Conv2D(192, 384, kernel_size=3, padding=1),
            ReLU(),
            Conv2D(384, 256, kernel_size=3, padding=1),
            ReLU(),
            Conv2D(256, 256, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=3, stride=2),
        )
        self.avgpool = AdaptiveAvgPool2D((6, 6))
        self.classifier = Sequential(
            Dropout(p=0.5),
            Linear(256 * 6 * 6, 4096),
            ReLU(),
            Dropout(p=0.5),
            Linear(4096, 4096),
            ReLU(),
            Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class OriginalAlexNet(Module):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1_1 = Conv2D(3, 48, kernel_size=11, stride=4, padding=2)
        self.relu1_1 = ReLU()
        self.pool1_1 = MaxPool2D(kernel_size=3, stride=2)

        self.conv2_1 = Conv2D(48, 128, kernel_size=5, padding=2)
        self.relu2_1 = ReLU()
        self.pool2_1 = MaxPool2D(kernel_size=3, stride=2)

        self.conv3_1 = Conv2D(256, 192, kernel_size=3, padding=1)
        self.relu3_1 = ReLU()

        self.conv4_1 = Conv2D(192, 192, kernel_size=3, padding=1)
        self.relu4_1 = ReLU()

        self.conv5_1 = Conv2D(192, 128, kernel_size=3, padding=1)
        self.relu5_1 = ReLU()
        self.pool5_1 = MaxPool2D(kernel_size=3, stride=2)

        self.conv1_2 = Conv2D(3, 48, kernel_size=11, stride=4, padding=2)
        self.relu1_2 = ReLU()
        self.pool1_2 = MaxPool2D(kernel_size=3, stride=2)

        self.conv2_2 = Conv2D(48, 128, kernel_size=5, padding=2)
        self.relu2_2 = ReLU()
        self.pool2_2 = MaxPool2D(kernel_size=3, stride=2)

        self.conv3_2 = Conv2D(256, 192, kernel_size=3, padding=1)
        self.relu3_2 = ReLU()

        self.conv4_2 = Conv2D(192, 192, kernel_size=3, padding=1)
        self.relu4_2 = ReLU()

        self.conv5_2 = Conv2D(192, 128, kernel_size=3, padding=1)
        self.relu5_2 = ReLU()
        self.pool5_2 = MaxPool2D(kernel_size=3, stride=2)

        self.avgpool = AdaptiveAvgPool2D((6, 6))

        self.dropout1 = Dropout(p=0.5)
        self.fc6 = Linear(256 * 6 * 6, 4096)
        self.relu6 = ReLU()

        self.dropout2 = Dropout(p=0.5)
        self.fc7 = Linear(4096, 4096)
        self.relu7 = ReLU()

        self.fc8 = Linear(4096, num_classes)

    def forward(self, x):
        x1_1 = self.pool1_1(self.relu1_1(self.conv1_1(x)))
        x1_2 = self.pool1_2(self.relu1_2(self.conv1_2(x)))

        x2_1 = self.pool2_1(self.relu2_1(self.conv2_1(x1_1)))
        x2_2 = self.pool2_2(self.relu2_2(self.conv2_2(x1_2)))

        x2 = concat((x2_1, x2_2), axis=1)

        x3_1 = self.relu3_1(self.conv3_1(x2))
        x3_2 = self.relu3_2(self.conv3_2(x2))

        x4_1 = self.relu4_1(self.conv4_1(x3_1))
        x4_2 = self.relu4_2(self.conv4_2(x3_2))

        x5_1 = self.pool5_1(self.relu5_1(self.conv5_1(x4_1)))
        x5_2 = self.pool5_2(self.relu5_2(self.conv5_2(x4_2)))

        x5 = concat((x5_1, x5_2), axis=1)

        x_avg = self.avgpool(x5)
        x_flat = x_avg.flatten(1)

        out = self.fc8(
            self.relu7(
                self.fc7(
                    self.dropout2(
                        self.relu6(
                            self.fc6(
                                self.dropout1(x_flat)
                            )
                        )
                    )
                )
            )
        )
        return out
