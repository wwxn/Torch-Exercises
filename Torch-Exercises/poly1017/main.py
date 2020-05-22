import torch
import definition as df
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
x=torch.FloatTensor(np.float32([i*0.01 for i in range(-200,200)]))
input=df.make_features(x)
target=df.f(input)
model=df.PolyLinear()
criterion=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=1e-3)
plt.plot(x,target,'ro')

flag=True
time=0
while flag:
    out=model.forward(input)
    loss=criterion.forward(out,target)
    # print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    time+=1
    if time%1000==0:
        print("current loss is{}".format(loss.data).title())
        print(list(model.parameters()))
        # print(model)
    if loss.data<1e-3:
        flag=False

model.eval()
print(list(model.parameters()))
out=model.forward(input)
plt.plot(x.detach().numpy(),out.detach().numpy())
plt.show()
#
# a=torch.Tensor([1,2,3])
# a.backward()
# print(a)