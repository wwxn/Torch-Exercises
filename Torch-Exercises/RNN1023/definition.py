import torch
import torch.nn as nn
import numpy as np


def normalize(data):
    data_out = [(x - np.mean(data)) / np.std(data) for x in data]
    return data_out


def make_data(data_list, input_number):
    input_list = []
    output_list = []
    batches = data_list.__len__() - input_number
    for i in range(batches):
        input_data = [data_list[i + n] for n in range(input_number)]
        output_list.append(data_list[i + input_number])
        input_list.append(input_data)
    return torch.Tensor(input_list).reshape((batches, -1, input_number)), torch.Tensor(output_list).unsqueeze(1)


class MyRNN(nn.Module):
    def __init__(self, input_number, hidden_number, output_number):
        super(MyRNN, self).__init__()
        self.layerRNN = nn.LSTM(input_size=input_number, hidden_size=hidden_number, num_layers=1, batch_first=False)
        self.layerFC = nn.Sequential(nn.Linear(hidden_number, output_number))

    def forward(self, x):
        x, _ = self.layerRNN.forward(x)
        # print(hstate)
        x = x.reshape(-1, x.size(-1))
        x = self.layerFC.forward(x)
        return x
