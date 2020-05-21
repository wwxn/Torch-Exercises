import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import myclass as mc
import torch.optim as optim
from torch.autograd import Variable
arrayx=[]

for time in range(1,21):
    arrayx.append([time])

arrayy=[]

for time in range(1,21):
    randnum=rd.random()
    if randnum>0.5:
        arrayy.append([float(time)])
    else:
        arrayy.append([float(time)])

x_train=torch.Tensor(arrayx)
y_train=torch.Tensor(arrayy)

print(x_train)

model=mc.LinearRegression()

criterion=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.0001)

num_epochs=1000
inputs=x_train
target=y_train
for epoch in range(num_epochs):

    out=model.forward(inputs)
    print(out)
    loss=criterion.forward(out,target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)%20==0:
        print("Epoch[{}/{}],loss:{:6f}".format(epoch+1,num_epochs,loss.data))

model.eval()
print(list(model.parameters()))
predict=model.forward(inputs)
plt.plot(inputs.detach().numpy(),target.detach().numpy(),'ro')
plt.plot(inputs.detach().numpy(),predict.detach().numpy())
plt.show()


