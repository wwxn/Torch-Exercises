import torch
import torch.nn as nn

BATCH_SIZE = 64

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_size, 256),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(256, 256),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(256, 1),
                                    nn.Sigmoid())

    def forward(self, x):
        x = self.linear(x)
        return x


class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_size, 256),
                                    nn.ReLU(True),
                                    nn.Linear(256, 256),
                                    nn.ReLU(True),
                                    nn.Linear(256, output_size),
                                    nn.Tanh())

    def forward(self, x):
        x = self.linear(x)
        return x

def measure_data(loaded_data):
    data=loaded_data[0]
    data=data.reshape(data.size(0),-1)
    return data
