import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.liner=nn.Linear(1,1)

    def forward(self,x):
        out=self.liner.forward(x)
        return out
