import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self, in_dim):
        super(Net, self).__init__()
        self.layer_1 = nn.Sequential(nn.Linear(in_dim, 300), nn.BatchNorm1d(300), nn.ReLU(True))
        self.layer_2 = nn.Sequential(nn.Linear(300, 100), nn.BatchNorm1d(100), nn.ReLU(True))
        self.output = nn.Sequential(nn.Linear(100, 10))

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.output(x)
        return x


# 训练
def train():
    train_loss = 0
    for data in train_load:
        # 每一次迭代都返回一组输入数据和标签
        img, label = data
        # 将图片进行image转换
        img = img.view(img.size(0), -1)
        # 构造变量
        img = Variable(img)
        label = Variable(label)
        # 获得模型的计算结果
        out = model(img)
        # 计算损失值
        loss = criterion(out, label)
        train_loss += loss.data
        # 梯度清零
        opt.zero_grad()
        # 损失值传播
        loss.backward()
        # 优化器优化
        opt.step()
    print('train_loss={:.6f}'.format((train_loss / len(train_load))), end=',')


# 测试
def test():
    correct = 0
    test_loss = 0
    for data in test_load:
        # 每一次迭代都返回一组输入数据和标签
        img, label = data
        # 将图片进行image转换
        img = img.view(img.size(0), -1)
        # 构造变量
        img = Variable(img)
        label = Variable(label)
        # 获得模型的结果
        out = model(img)
        # 计算损失值
        loss = criterion(out, label)
        test_loss += loss
        # 计算准确率
        _, predict = torch.max(out, 1)
        equal = predict == label
        correct += torch.sum(equal)
    print('test_loss:{:.6f}'.format(test_loss / len(test_load)), end=',')
    print('accuracy:{:.2f}%'.format(correct / len(test_load)))


if __name__ == '__main__':
    # 导入数据集
    # 训练集
    train_data = datasets.MNIST(root='./',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)
    # 测试集
    test_data = datasets.MNIST(root='./',
                               train=False,
                               transform=transforms.ToTensor(),
                               download=True)
    # 数据装载
    # 批次大小
    batch_size = 100
    # 装载训练集，将数据打乱
    train_load = DataLoader(dataset=train_data,
                            batch_size=batch_size,
                            shuffle=True)
    # 装载测试集，将数据打乱
    test_load = DataLoader(dataset=test_data,
                           batch_size=batch_size,
                           shuffle=True)

    # 构建模型
    # 网络的结构输入为28*28，第一层隐藏层的输出为300，第二层隐藏层的输出为100，最后输出为10
    model = Net(28 * 28)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.5)

    # 训练和预测
    train()
    test()
