import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import definition as df

IMG_SIZE = 28 * 28
VECTOR_SIZE = 3
BATCH_SIZE = 1

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])

train_dataset = datasets.MNIST(root='.\data', train=True, transform=transform, download=True)
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
datas = [data for data in train_data_loader]

model_D = df.Discriminator(IMG_SIZE)
model_G = df.Generator(VECTOR_SIZE, IMG_SIZE)

D_target = torch.ones(BATCH_SIZE).reshape(BATCH_SIZE, -1)
G_target = torch.zeros(BATCH_SIZE).reshape(BATCH_SIZE, -1)

criterion = nn.BCELoss()
D_optimizer = torch.optim.Adam(model_D.parameters(), 0.0003)
G_optimizer = torch.optim.Adam(model_G.parameters(), 0.0003)

for i in range(20):
    G_input = torch.randn((BATCH_SIZE, VECTOR_SIZE))
    D_input = df.measure_data(datas[random.randint(0, 63)])
    D_out = model_D.forward(D_input)
    # print(D_input[0])
    G_out = model_G.forward(G_input)
    G_out = model_D.forward(G_out)
    D_loss = criterion.forward(D_out, D_target)
    G_loss = criterion.forward(G_out, G_target)
    total_loss = D_loss + G_loss
    D_optimizer.zero_grad()
    total_loss.backward()
    D_optimizer.step()
    # print(total_loss.data)

for i in range(500):
    G_input = torch.randn((BATCH_SIZE, VECTOR_SIZE))
    G_out = model_G.forward(G_input)
    G_out = model_D.forward(G_out)
    G_loss = criterion.forward(G_out, D_target)
    G_optimizer.zero_grad()
    G_loss.backward()
    G_optimizer.step()
    # print(G_loss.data)

print('Training Finished')
torch.save(model_D.state_dict(),'model_D.pt')
torch.save(model_G.state_dict(),'model_G.pt')

model_G.eval()
G_input = torch.randn((1, VECTOR_SIZE))
test_out = model_G.forward(G_input)

unloader = transforms.ToPILImage()
test_out = test_out.data.reshape(1, 28, 28)
# test_out=test_out.ge(-0.5).float()
print(test_out)
image = unloader(test_out)
plt.imshow(image,'gray')
plt.show()
