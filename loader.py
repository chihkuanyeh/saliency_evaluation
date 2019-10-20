import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np


def mnist_loaders(batch_size):
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, pin_memory=False)
    return train_loader, test_loader


def mnist_load_model(path, state_dict=False, tf=False):
    if state_dict:
        if tf:
            print('Loading from tf weight...')
            model = Mnist_model()
            tfmodel = np.load(path)
            weight_list = ['conv1', 'biasc1', 'conv2', 'biasc2', 'fc1', 'biasf1', 'fc2', 'biasf2']
            tffile = [tfmodel[index] for index in weight_list]
            for count, k in enumerate(model.state_dict().keys()):
                size = len(model.state_dict()[k].numpy().shape)
                if size == 4:
                    model.state_dict()[k].copy_(torch.from_numpy(tffile[count].transpose(3, 2, 0, 1)))
                elif size == 2:
                    model.state_dict()[k].copy_(torch.from_numpy(tffile[count].transpose(1, 0)))
                elif size == 1:
                    model.state_dict()[k].copy_(torch.from_numpy(tffile[count]))
        else:
            model = Mnist_model()
            model.load_state_dict(torch.load(path))
    else:
        model = torch.load(path)
    return model


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = Parameter(torch.Tensor(
                out_channels, in_channels, kernel_size, kernel_size))

        self.bias = Parameter(torch.Tensor(out_channels))

    def forward(self, x):

        pad = self.kernel_size-1
        if pad % 2 == 0:
            pad_val = pad // 2
            padding = (pad_val, pad_val, pad_val, pad_val)
        else:
            pad_val_start = pad // 2
            pad_val_end = pad - pad_val_start
            padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end)

        x_torch = F.pad(x, padding, "constant", 0)
        return F.conv2d(x_torch, self.weight, padding=0, stride=1, bias=self.bias)


class Mnist_model(nn.Module):
    def __init__(self):
        super(Mnist_model, self).__init__()
        self.conv1 = Conv2dSame(1, 32, 5)
        self.conv2 = Conv2dSame(32, 64, 5)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        y = x.permute(0, 2, 3, 1)
        x = y.contiguous().view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
