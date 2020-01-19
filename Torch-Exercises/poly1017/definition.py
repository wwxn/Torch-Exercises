import torch
import random
import torch.nn as nn

def make_features(x):
    x=x.unsqueeze(1)
    return torch.cat([x**i for i in range(1,4)],1)

def f(x) :
    w = torch.Tensor([3,-2,1])
    b = torch.Tensor([5])
    w = w.unsqueeze(1)
    return x.mm(w)+b[0]

class PolyLinear(nn.Module):
    def __init__(self):
        super(PolyLinear, self).__init__()
        self.linear=nn.Linear(3,1)

    def forward(self,x):
        return self.linear.forward(x)