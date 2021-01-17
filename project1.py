# -*- coding: UTF-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


def get_data():
    path = "/Users/yeeland/MLdata/iris.data"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    df = pd.read_csv(path, sep=',', names=names)
    return df.iloc[:, 0:4], df.iloc[:, 4]


if __name__ == '__main__':
    # 获取数据
    trainDF = pd.DataFrame()
    X, y = get_data()
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    # 训练模型
    model = svm.SVC()
    model.fit(x_train, y_train)
    # 预测
    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_predict, y_test)
    print('accuracy:', accuracy)
