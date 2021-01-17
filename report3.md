## 实验报告3：字符识别
### 实验目标
MNIST手写识别数据集：http://yann.lecun.com/exdb/mnist/

任务:识别字符

### 实验思路
ML算法：神经网络

### 实验过程

1. 导入数据集。使用torchvision.datasets包中的MNIST()方法可以直接导入MNIST数据集。
        
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
2. 装载数据。设置批次大小为100，分别装载训练集和测试集，并将数据打乱。

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
3. 构建模型。设置网络的输入结构为28*28，第一层隐藏层的输出为300，第二层隐藏层的输出为100，最后输出为10。

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
        
        model = Net(28 * 28)
4. 定义损失函数和优化器。

        criterion = nn.CrossEntropyLoss()
        opt = optim.SGD(model.parameters(), lr=0.5)

5. 训练模型并输出训练损失值。

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
    
6. 预测测试集并输出预测准确值。

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
        
### 实验结果
    
![avatar](/Users/yeeland/MLdata/pro3.png)

如图所示，运行得到的训练损失值为0.180509，测试的损失值0.102898，模型预测准确度为96.80%。