from torchvision import datasets,transforms
from torch.utils.data import DataLoader
data_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
train_dataset=datasets.MNIST(root='./data',train=True,transform=data_tf,download=True)
test_dataset=datasets.MNIST(root='./data',train=False,transform=data_tf)

trainLoader=DataLoader(train_dataset,1,shuffle=True)

for data in trainLoader:
    img,label=data
    img=img.view(img.size(0),-1)
    print('{},{}'.format(img,label))