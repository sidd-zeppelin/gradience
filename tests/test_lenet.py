import torch
import torch.nn as nn
import numpy as np
from gradience.tensor import Tensor
from gradience.nn.models.lenet import LeNet5

class PtLeNet5(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc3 = nn.Linear(16 * 5 * 5, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.pool(self.tanh(self.conv1(x)))
        x = self.pool(self.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.tanh(self.fc3(x))
        x = self.tanh(self.fc4(x))
        x = self.fc5(x)
        return x

def copy_weights_lenet(pt_model, gr_model):
    convs = [('conv1', 'conv1'), ('conv2', 'conv2')]
    for pt_name, gr_name in convs:
        pt_conv = getattr(pt_model, pt_name)
        gr_conv = getattr(gr_model, gr_name)
        gr_conv.weight.data[...] = pt_conv.weight.cpu().detach().numpy()
        if gr_conv.bias is not None:
            gr_conv.bias.data[...] = pt_conv.bias.cpu().detach().numpy()

    linears = [('fc3', 'fc3'), ('fc4', 'fc4'), ('fc5', 'fc5')]
    for pt_name, gr_name in linears:
        pt_linear = getattr(pt_model, pt_name)
        gr_linear = getattr(gr_model, gr_name)
        gr_linear.weight.data[...] = pt_linear.weight.cpu().detach().numpy().T
        if gr_linear.bias is not None:
            gr_linear.bias.data[...] = pt_linear.bias.cpu().detach().numpy()

def test_lenet_forward_backward_gray():
    run_lenet_test(1)

def test_lenet_forward_backward_rgb():
    run_lenet_test(3)

def run_lenet_test(in_channels):
    x_data = np.random.randn(2, in_channels, 32, 32).astype(np.float64)
    y_data = np.zeros((2, 10))
    y_data[0, 3] = 1.0
    y_data[1, 7] = 1.0

    gr_model = LeNet5(in_channels=in_channels, num_classes=10)
    pt_model = PtLeNet5(in_channels=in_channels, num_classes=10).double()
    copy_weights_lenet(pt_model, gr_model)

    gr_model.eval()
    pt_model.eval()

    x_pt = torch.tensor(x_data, requires_grad=True, dtype=torch.float64)
    y_pt = torch.tensor(y_data, dtype=torch.float64)

    x_gr = Tensor(x_data, requires_grad=True)
    y_gr = Tensor(y_data)

    out_pt = pt_model(x_pt)
    loss_pt = ((out_pt - y_pt) ** 2).mean()
    loss_pt.backward()

    out_gr = gr_model(x_gr)
    loss_gr = ((out_gr - y_gr) ** 2).mean()
    loss_gr.backward()

    assert np.allclose(out_gr.data, out_pt.detach().numpy(), atol=1e-12)
    assert np.allclose(loss_gr.item(), loss_pt.item(), atol=1e-12)
    assert np.allclose(x_gr.grad, x_pt.grad.numpy(), atol=1e-12)

    for pt_name, gr_name in [('conv1', 'conv1'), ('conv2', 'conv2')]:
        pt_conv = getattr(pt_model, pt_name)
        gr_conv = getattr(gr_model, gr_name)
        assert np.allclose(gr_conv.weight.grad, pt_conv.weight.grad.numpy(), atol=1e-12)
        assert np.allclose(gr_conv.bias.grad, pt_conv.bias.grad.numpy(), atol=1e-12)

    for pt_name, gr_name in [('fc3', 'fc3'), ('fc4', 'fc4'), ('fc5', 'fc5')]:
        pt_linear = getattr(pt_model, pt_name)
        gr_linear = getattr(gr_model, gr_name)
        assert np.allclose(gr_linear.weight.grad, pt_linear.weight.grad.numpy().T, atol=1e-12)
        assert np.allclose(gr_linear.bias.grad, pt_linear.bias.grad.numpy(), atol=1e-12)
