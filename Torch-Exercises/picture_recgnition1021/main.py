import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import definition as df
import torch.nn as nn
import torch.optim as optim

# root="pic\dogs"
# save_root="train_pic\dogs"
# df.measure_pic(root,save_root)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_dataset = datasets.ImageFolder(root='train_pic', transform=transform)
train_dataload = DataLoader(train_dataset, 30, shuffle=True)
train_data = [data for data in train_dataload]
train_data = train_data[0]
train_data_input = train_data[0]
train_data_target = train_data[1]

transform = transforms.Compose(
    [transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
test_dataset = datasets.ImageFolder(root='test_pic', transform=transform)
test_dataload = DataLoader(test_dataset, 30, shuffle=False)
test_data = [data for data in test_dataload]
test_data = test_data[0]
test_data_input = test_data[0]
test_data_target = test_data[1]

model = df.myModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=1)

flag = True
time = 0
while flag:
    time += 1
    out = model.forward(train_data_input)
    # print (out)
    loss = criterion.forward(out, train_data_target)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    rate = df.calculate_rate(out, train_data_target)
    print(rate)
    if time > 100 and rate == 1:
        flag = False
model.eval()
output = model.forward(test_data_input)
print(output)
print(test_data_target)

rate = df.calculate_rate(output, test_data_target)
print(rate)
