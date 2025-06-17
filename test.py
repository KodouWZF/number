import PIL.Image
import torch
import random
import PIL
import numpy as np

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F


batch_size = 64

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transform, download=True)  # train=True训练集，=False测试集
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）
    
model = torch.load("./models/model_Mnist_10.pth", weights_only=False)  # 加载模型


fig = plt.figure()
for i in range(12):
    random_number = random.randint(0, len(test_dataset) - 1)
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    plt.imshow(test_dataset.data[random_number], cmap='gray', interpolation='none')
    #print(test_dataset.data[i].shape)
    test = test_dataset.data[random_number].float().reshape(1, 1, 28, 28)
    #print(test.shape)
    pre = model(test_dataset.data[random_number].float().reshape(1, 1, 28, 28)) 
    plt.title("forecast: {}".format(torch.argmax(pre).item()))
    plt.xticks([])
    plt.yticks([])
plt.show()