import torch
import torch.nn as nn
def read_data(path):
    with open(path) as f:
        datax = f.readlines()
        datay = []
        number = datax.__len__()
        datax = [i.split('\n') for i in datax]
        datax = [i[0].split(',') for i in datax]
        for i in range(number):
            for j in range(3):
                datax[i][j] = float(datax[i][j])
        for i in range(number):
            datay.append(datax[i].pop())
        datax = torch.Tensor(datax)
        datay = torch.Tensor(datay)
        datay = datay.unsqueeze(1)
        return datax,datay

class Classify(nn.Module):
    def __init__(self):
        super(Classify, self).__init__()
        self.linear=nn.Linear(2,1)
        self.sig=nn.Sigmoid()

    def forward(self,x):
        y=self.linear.forward(x)
        y=self.sig.forward(y)
        return y