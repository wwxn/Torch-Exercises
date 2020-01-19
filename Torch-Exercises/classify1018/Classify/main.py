import definition as df
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
datax,datay=df.read_data("data.txt")
print(datax)
print(datay)
model=df.Classify()
criterion=nn.BCELoss()
optimizer=optim.SGD(model.parameters(),lr=0.1)
flag=True
mask=[]
time=0
while flag:
    time+=1
    out=model.forward(datax)
    loss=criterion.forward(out,datay)
    correct_rate=out.ge(0.5).float()
    correct_rate=correct_rate==datay
    correct_rate=correct_rate.sum()/20.0
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if time%1000==0:
        print('Loss is {},correct rate is {}'.format(loss.data,correct_rate).title())
    if correct_rate==1 and loss<0.1:
        print('Loss is {},correct rate is {}'.format(loss.data, correct_rate).title())
        flag=False

model.eval()
print(model.forward(torch.Tensor([[50,3]])))
print(model.forward(torch.Tensor([[0,3]])))