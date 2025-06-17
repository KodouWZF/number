import torch
import numpy as np
import torch.nn.functional as F

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


#超参数设置
batch_size = 64             #批量大小
learning_rate = 0.01        #学习率
momentum = 0.5              #动量
EPOCH = 10                  #训练轮数

#数据预处理方法定义
transform = transforms.Compose([
    transforms.ToTensor(),                              # 转为张量
    transforms.Normalize((0.1307,), (0.3081,))          # 归一化: (mean : 0.1307, std : 0.3081)
    ]
)

# MNIST数据集下载和加载
train_dataset = datasets.MNIST(
    root='./data/mnist',        # 数据集路径
    train=True,                 # 是否为训练集：True表示是训练集，False表示是测试集
    transform=transform,        # 数据预处理方法
    download=True               # 是否下载数据集(如果本地没有的话自动下载)
)

test_dataset = datasets.MNIST(
    root='./data/mnist',        # 数据集路径
    train=False,                # 是否为训练集：True表示是训练集，False表示是测试集
    transform=transform,        # 数据预处理方法
    download=True               # 是否下载数据集(如果本地没有的话自动下载
)

print("训练集大小:", len(train_dataset))  # 打印训练集大小
print("测试集大小:", len(test_dataset))    # 打印测试集大小

# 数据加载器
train_loader = DataLoader(
    train_dataset,              # 数据集
    batch_size=batch_size,      # 批量大小
    shuffle=True,               # 是否打乱数据集顺序：True表示打乱，False表示不打乱
    num_workers=0,              # 多线程加载数据
    drop_last=True              # 是否丢弃最后一个不完整的batch: True表示丢弃，False表示保留
)

test_loader = DataLoader(
    test_dataset,               # 数据集
    batch_size=batch_size,      # 批量大小
    shuffle=False,              # 是否打乱数据集顺序：True表示打乱，False表示不打乱
    num_workers=0,              # 多线程加载数据
    drop_last=True              # 是否丢弃最后一个不完整的batch: True表示丢弃，False表示保留
)

# 可视化训练集数据
fig = plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')
    plt.title("Labels: {}".format(train_dataset.train_labels[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

# 训练集乱序，测试集有序
# Design model using class ------------------------------------------------------------------------------
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


model = Net()


# Construct loss and optimizer ------------------------------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量



# Train and Test CLASS --------------------------------------------------------------------------------------
# 把单独的一轮一环封装在函数类里
def train(epoch):
    running_loss = 0.0  # 这整个epoch的loss清零
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        # 把运行中的loss累加起来，为了下面300次一除
        running_loss += loss.item()
        # 把运行中的准确率acc算出来
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 300 == 299:  # 不想要每一次都出loss，浪费时间，选择每300次出一个平均损失,和准确率
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0  # 这小批300的loss清零
            running_total = 0
            running_correct = 0  # 这小批300的acc清零

    torch.save(model, f'./models/model_Mnist_{epoch+1}.pth')
    torch.save(optimizer.state_dict(), f'./optimizers/optimizer_Mnist_{epoch+1}.pth')

def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            total += labels.size(0)  # 张量之间的比较运算
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch+1, EPOCH, 100 * acc))  # 求测试的准确率，正确数/总数
    return acc

# Start train and Test --------------------------------------------------------------------------------------
if __name__ == '__main__':
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch)
        # if epoch % 10 == 9:  #每训练10轮 测试1次
        acc_test = test()
        acc_list_test.append(acc_test)

    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.show()