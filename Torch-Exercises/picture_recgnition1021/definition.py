from PIL import Image
import torch.nn as nn


def measure_pic(root, save_root):
    for pic_num in range(1, 15 + 1):
        pic_name = '\\' + str(pic_num) + '.jpg'
        pic_path = root + pic_name
        pic = Image.open(pic_path)
        pic = pic.resize((128, 128))
        save_path = save_root + '\\0' + str(pic_num) + '.jpg'
        pic.save(save_path)


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            nn.Linear(25088, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def calculate_rate(input_data, target):
    mask = []
    input_data = input_data.detach().numpy()
    for item in input_data:
        item = list(item)
        max_index = item.index(max(item))
        mask.append(max_index)
    rate = (mask == target.detach().numpy()).sum() / mask.__len__()
    return rate
