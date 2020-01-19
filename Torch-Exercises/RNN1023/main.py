import torch
import definition as df
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from torchvision import transforms
h_state = None
datas = [(i-250)**2 for i in range(500)]
data=df.normalize(datas)
input_data, output_data = df.make_data(data, 2)
x = [i for i in range(200)]
model = df.MyRNN(input_number=2, hidden_number=50, output_number=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
time = 0
plt.plot(data,'r-')



for i in range(100):
    time += 1
    out = model.forward(input_data)
    print(out)
    loss = criterion.forward(out, output_data)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    print(loss.data)
model.eval()
plt.plot(model.forward(input_data).detach().numpy())
plt.show()
