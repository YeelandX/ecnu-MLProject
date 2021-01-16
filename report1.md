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

1. 从网站上下载数据集保存至本地path路径下。
通过查看该数据集可以发现该数据集有四个类别分别是
'sepal-length', 'sepal-width', 'petal-length', 'petal-width'。
调用pandas包中read_csv方法读取训练数据和标签。

        path = "/Users/yeeland/MLdata/iris.data"
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        df = pd.read_csv(path, sep=',', names=names)
2. 
3.





### 实验结果

