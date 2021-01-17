## 实验报告1
### 实验目标
UCI数据集: http://archive.ics.uci.edu/ml/index.php

任务:
任选一个数据集，
任选一种ML算法：逻辑回归、决策树、神经网络、SVM等。

### 实验思路
数据集：Iris数据集

ML算法：SVM

### 实验过程

1. 获取数据集。从网站上下载数据集保存至本地path路径下。
通过查看该数据集可以发现该数据集有四个类别分别是
'sepal-length', 'sepal-width', 'petal-length', 'petal-width'。
调用pandas包中read_csv方法读取训练数据和标签。

        # 获取数据
        path = "/Users/yeeland/MLdata/iris.data"
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        df = pd.read_csv(path, sep=',', names=names)
2. 划分测试集和训练集，选取数据集中25%作为测试

        # 划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
3. 使用sklearn.svm包中的SVC()方法进行训练

        # 训练模型
        model = svm.SVC()
        model.fit(x_train, y_train)
4. 预测模型，使用训练后的模型对测试集进行预测，
并使用sklean.metrics包中的accuracy_score()方法展示训练准确率。

        # 预测
        y_predict = model.predict(x_test)
        accuracy = accuracy_score(y_predict, y_test)
        print('accuracy:', accuracy)

### 实验结果

![avatar](/Users/yeeland/MLdata/pro1.png)

如图所示，运行得到的训练准确率约为0.95。