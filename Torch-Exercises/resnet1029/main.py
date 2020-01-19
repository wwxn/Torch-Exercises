import random

import torch.nn as nn
import torch.optim as optim

import definition as df
import torchvision.models as models
train_data, test_data = df.create_data()

data_loaded = [item for item in train_data]
data = data_loaded[0]

inputs = data[0]
# inputs = inputs.view(inputs.size(0), -1)
target = data[1]
# model = df.SimpleNet(28 * 28, 100, 30, 10)
model=models.resnet18()
model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc=nn.Linear(in_features=512, out_features=10, bias=True)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)

time = 0
flag = True
while flag:
    data = data_loaded[random.randint(0, 63)]
    inputs = data[0]
    # inputs = inputs.view(inputs.size(0), -1)
    target = data[1]
    time += 1
    out = model.forward(inputs)
    loss = criterion.forward(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    rate = df.calculate_rate(out, target)
    # if time % 1000 == 0:
    print("correct rate is {},loss is {}".format(rate, loss).title())
    if loss < 0.01:
        flag = False

model.eval()
data_for_test = [item for item in test_data]
data_for_test = data_for_test[0]
input_for_test = data_for_test[0]
# input_for_test = input_for_test.view(input_for_test.size(0), -1)
target_for_test = data_for_test[1]
out = model.forward(input_for_test)
answer_out = df.predict(out)
correct_rate = df.calculate_rate(out, target_for_test)
print(
    'The correct answer is {},and the output is {}.Correct rate is{}'.format(target_for_test, answer_out, correct_rate))
