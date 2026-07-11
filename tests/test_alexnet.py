import pytest
import numpy as np
import torch
import torch.nn as nn
from gradience.tensor import Tensor
from gradience.nn.models.alexnet import AlexNet


class PtAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def copy_weights(pt_model, gr_model):
    # Copy features conv weights
    conv_indices = [0, 3, 6, 8, 10]
    for idx in conv_indices:
        pt_conv = pt_model.features[idx]
        gr_conv = gr_model.features[idx]
        gr_conv.weight.data[...] = pt_conv.weight.detach().numpy()
        if gr_conv.bias is not None:
            gr_conv.bias.data[...] = pt_conv.bias.detach().numpy()

    # Copy classifier linear weights
    linear_indices = [1, 4, 6]
    for idx in linear_indices:
        pt_linear = pt_model.classifier[idx]
        gr_linear = gr_model.classifier[idx]
        gr_linear.weight.data[...] = pt_linear.weight.detach().numpy().T
        if gr_linear.bias is not None:
            gr_linear.bias.data[...] = pt_linear.bias.detach().numpy()


def test_alexnet_parameter_count():
    model = AlexNet(num_classes=10)
    parameters = model.parameters()
    # 5 conv layers (weight, bias) = 10 parameters
    # 3 linear layers (weight, bias) = 6 parameters
    # Total = 16 parameters
    assert len(parameters) == 16


def test_alexnet_forward_backward_match_pytorch():
    # Set seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    num_classes = 10
    batch_size = 1
    channels = 3
    height, width = 64, 64

    # Instantiate both models
    gr_model = AlexNet(num_classes=num_classes)
    pt_model = PtAlexNet(num_classes=num_classes).double()

    # Copy weights from PyTorch to Gradience
    copy_weights(pt_model, gr_model)

    # Put both models in eval mode to avoid dropout randomness
    gr_model.eval()
    pt_model.eval()

    # Input tensor
    x_np = np.random.randn(batch_size, channels, height, width).astype(np.float64)
    x_pt = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)
    x_gr = Tensor(x_np, requires_grad=True)

    # Forward pass
    out_pt = pt_model(x_pt)
    out_gr = gr_model(x_gr)

    assert np.allclose(out_gr.data, out_pt.detach().numpy(), atol=1e-7, rtol=1e-5)

    # Backward pass
    loss_pt = out_pt.sum()
    loss_gr = out_gr.sum()

    loss_pt.backward()
    loss_gr.backward()

    # Check input gradients
    assert np.allclose(x_gr.grad, x_pt.grad.numpy(), atol=1e-7, rtol=1e-5)

    # Check parameter gradients
    conv_indices = [0, 3, 6, 8, 10]
    for idx in conv_indices:
        pt_conv = pt_model.features[idx]
        gr_conv = gr_model.features[idx]
        assert np.allclose(gr_conv.weight.grad, pt_conv.weight.grad.numpy(), atol=1e-6, rtol=1e-4)
        if gr_conv.bias is not None:
            assert np.allclose(gr_conv.bias.grad, pt_conv.bias.grad.numpy(), atol=1e-6, rtol=1e-4)

    linear_indices = [1, 4, 6]
    for idx in linear_indices:
        pt_linear = pt_model.classifier[idx]
        gr_linear = gr_model.classifier[idx]
        assert np.allclose(gr_linear.weight.grad, pt_linear.weight.grad.numpy().T, atol=1e-6, rtol=1e-4)
        if gr_linear.bias is not None:
            assert np.allclose(gr_linear.bias.grad, pt_linear.bias.grad.numpy(), atol=1e-6, rtol=1e-4)


class PtOriginalAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2)
        self.conv2_1 = nn.Conv2d(48, 128, kernel_size=5, padding=2)
        self.conv3_1 = nn.Conv2d(256, 192, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(192, 128, kernel_size=3, padding=1)

        self.conv1_2 = nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2)
        self.conv2_2 = nn.Conv2d(48, 128, kernel_size=5, padding=2)
        self.conv3_2 = nn.Conv2d(256, 192, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(192, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1_1 = self.pool(self.relu(self.conv1_1(x)))
        x1_2 = self.pool(self.relu(self.conv1_2(x)))

        x2_1 = self.pool(self.relu(self.conv2_1(x1_1)))
        x2_2 = self.pool(self.relu(self.conv2_2(x1_2)))

        x2 = torch.cat((x2_1, x2_2), dim=1)

        x3_1 = self.relu(self.conv3_1(x2))
        x3_2 = self.relu(self.conv3_2(x2))

        x4_1 = self.relu(self.conv4_1(x3_1))
        x4_2 = self.relu(self.conv4_2(x3_2))

        x5_1 = self.pool(self.relu(self.conv5_1(x4_1)))
        x5_2 = self.pool(self.relu(self.conv5_2(x4_2)))

        x5 = torch.cat((x5_1, x5_2), dim=1)

        x_avg = self.avgpool(x5)
        x_flat = torch.flatten(x_avg, 1)

        x_fc6 = self.relu(self.fc6(x_flat))
        x_fc7 = self.relu(self.fc7(x_fc6))
        out = self.fc8(x_fc7)
        return out


def copy_weights_original(pt_model, gr_model):
    convs = [
        ('conv1_1', 'conv1_1'), ('conv1_2', 'conv1_2'),
        ('conv2_1', 'conv2_1'), ('conv2_2', 'conv2_2'),
        ('conv3_1', 'conv3_1'), ('conv3_2', 'conv3_2'),
        ('conv4_1', 'conv4_1'), ('conv4_2', 'conv4_2'),
        ('conv5_1', 'conv5_1'), ('conv5_2', 'conv5_2')
    ]
    for pt_name, gr_name in convs:
        pt_conv = getattr(pt_model, pt_name)
        gr_conv = getattr(gr_model, gr_name)
        gr_conv.weight.data[...] = pt_conv.weight.detach().numpy()
        if gr_conv.bias is not None:
            gr_conv.bias.data[...] = pt_conv.bias.detach().numpy()

    linears = [
        ('fc6', 'fc6'), ('fc7', 'fc7'), ('fc8', 'fc8')
    ]
    for pt_name, gr_name in linears:
        pt_linear = getattr(pt_model, pt_name)
        gr_linear = getattr(gr_model, gr_name)
        gr_linear.weight.data[...] = pt_linear.weight.detach().numpy().T
        if gr_linear.bias is not None:
            gr_linear.bias.data[...] = pt_linear.bias.detach().numpy()


def test_original_alexnet_forward_backward_match_pytorch():
    from gradience.nn.models.alexnet import OriginalAlexNet
    np.random.seed(42)
    torch.manual_seed(42)

    num_classes = 10
    batch_size = 1
    channels = 3
    height, width = 64, 64

    gr_model = OriginalAlexNet(num_classes=num_classes)
    pt_model = PtOriginalAlexNet(num_classes=num_classes).double()

    copy_weights_original(pt_model, gr_model)

    gr_model.eval()
    pt_model.eval()

    x_np = np.random.randn(batch_size, channels, height, width).astype(np.float64)
    x_pt = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)
    x_gr = Tensor(x_np, requires_grad=True)

    out_pt = pt_model(x_pt)
    out_gr = gr_model(x_gr)

    assert np.allclose(out_gr.data, out_pt.detach().numpy(), atol=1e-7, rtol=1e-5)

    loss_pt = out_pt.sum()
    loss_gr = out_gr.sum()

    loss_pt.backward()
    loss_gr.backward()

    assert np.allclose(x_gr.grad, x_pt.grad.numpy(), atol=1e-7, rtol=1e-5)

    convs = [
        ('conv1_1', 'conv1_1'), ('conv1_2', 'conv1_2'),
        ('conv2_1', 'conv2_1'), ('conv2_2', 'conv2_2'),
        ('conv3_1', 'conv3_1'), ('conv3_2', 'conv3_2'),
        ('conv4_1', 'conv4_1'), ('conv4_2', 'conv4_2'),
        ('conv5_1', 'conv5_1'), ('conv5_2', 'conv5_2')
    ]
    for pt_name, gr_name in convs:
        pt_conv = getattr(pt_model, pt_name)
        gr_conv = getattr(gr_model, gr_name)
        assert np.allclose(gr_conv.weight.grad, pt_conv.weight.grad.numpy(), atol=1e-6, rtol=1e-4)
        if gr_conv.bias is not None:
            assert np.allclose(gr_conv.bias.grad, pt_conv.bias.grad.numpy(), atol=1e-6, rtol=1e-4)

    linears = [
        ('fc6', 'fc6'), ('fc7', 'fc7'), ('fc8', 'fc8')
    ]
    for pt_name, gr_name in linears:
        pt_linear = getattr(pt_model, pt_name)
        gr_linear = getattr(gr_model, gr_name)
        assert np.allclose(gr_linear.weight.grad, pt_linear.weight.grad.numpy().T, atol=1e-6, rtol=1e-4)
        if gr_linear.bias is not None:
            assert np.allclose(gr_linear.bias.grad, pt_linear.bias.grad.numpy(), atol=1e-6, rtol=1e-4)


def test_concat_op():
    from gradience.tensor import concat
    a_np = np.random.randn(2, 3, 4).astype(np.float64)
    b_np = np.random.randn(2, 2, 4).astype(np.float64)

    a_gr = Tensor(a_np, requires_grad=True)
    b_gr = Tensor(b_np, requires_grad=True)

    a_pt = torch.tensor(a_np, requires_grad=True, dtype=torch.float64)
    b_pt = torch.tensor(b_np, requires_grad=True, dtype=torch.float64)

    y_gr = concat((a_gr, b_gr), axis=1)
    y_pt = torch.cat((a_pt, b_pt), dim=1)

    assert np.allclose(y_gr.data, y_pt.detach().numpy())

    loss_gr = y_gr.sum()
    loss_pt = y_pt.sum()

    loss_gr.backward()
    loss_pt.backward()

    assert np.allclose(a_gr.grad, a_pt.grad.numpy())
    assert np.allclose(b_gr.grad, b_pt.grad.numpy())

