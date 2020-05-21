from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn


def create_data():
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST('./data', train=True, transform=data_transform, download=True)
    test_dataset = datasets.MNIST('./data', transform=data_transform)
    train_loader = DataLoader(train_dataset, 64, shuffle=True)
    test_loader = DataLoader(test_dataset, 1000, shuffle=False)
    return train_loader, test_loader


class SimpleNet(nn.Module):
    def __init__(self, inputs, hidden_1, hidden_2, out):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv1', nn.Conv2d(1, 16, 3, padding=1))
        self.conv1.add_module('normalize2d1',nn.BatchNorm2d(16))
        self.conv1.add_module('relu1', nn.ReLU())
        self.conv1.add_module('pool1', nn.MaxPool2d((2, 2)))

        self.conv2 = nn.Sequential()
        self.conv2.add_module('conv2', nn.Conv2d(16, 32, 3, padding=1))
        self.conv2.add_module('normalize2d2', nn.BatchNorm2d(32))
        self.conv2.add_module('relu2', nn.ReLU())
        self.conv2.add_module('pool2', nn.MaxPool2d((2, 2)))

        self.full_connect = nn.Sequential()

        self.full_connect.add_module('linear1', nn.Linear(7 * 7 * 32, 100))
        self.full_connect.add_module('normalize1', nn.BatchNorm1d(100))
        self.full_connect.add_module('sigmoid1', nn.ReLU())

        self.full_connect.add_module('linear2', nn.Linear(100, 28))
        self.full_connect.add_module('normalize2', nn.BatchNorm1d(28))
        self.full_connect.add_module('sigmoid2', nn.ReLU())

        self.full_connect.add_module('linear3', nn.Linear(28, 10))
        self.full_connect.add_module('normalize3', nn.BatchNorm1d(10))
        self.full_connect.add_module('sigmoid3', nn.ReLU())

    def forward(self, x):
        y = self.conv1.forward(x)
        y = self.conv2.forward(y)
        y = y.view(y.size(0), -1)
        y = self.full_connect.forward(y)
        return y


def calculate_rate(input_data, target):
    mask = []
    input_data = input_data.detach().numpy()
    for item in input_data:
        item = list(item)
        max_index = item.index(max(item))
        mask.append(max_index)
    rate = (mask == target.detach().numpy()).sum() / mask.__len__()
    return rate


def predict(input_data):
    mask = []
    input_data = input_data.detach().numpy()
    for item in input_data:
        item = list(item)
        max_index = item.index(max(item))
        mask.append(max_index)
    return mask
