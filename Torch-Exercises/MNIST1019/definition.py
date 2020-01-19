import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_data():
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST('./data', train=True, transform=data_transform, download=True)
    test_dataset = datasets.MNIST('./data', transform=data_transform)
    train_loader = DataLoader(train_dataset, 64, shuffle=True)
    test_loader = DataLoader(test_dataset, 64, shuffle=False)
    return train_loader, test_loader


class SimpleNet(nn.Module):
    def __init__(self, inputs, hidden_1, hidden_2, out):
        super(SimpleNet, self).__init__()
        self.layer_1 = nn.Sequential(nn.Linear(inputs, hidden_1), nn.BatchNorm1d(hidden_1), nn.ReLU())
        self.layer_2 = nn.Sequential(nn.Linear(hidden_1, hidden_2), nn.BatchNorm1d(hidden_2), nn.ReLU())
        self.layer_3 = nn.Sequential(nn.Linear(hidden_2, out), nn.BatchNorm1d(out), nn.ReLU())

    def forward(self, x):
        y = self.layer_1.forward(x)
        y = self.layer_2.forward(y)
        y = self.layer_3.forward(y)
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
