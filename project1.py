# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

path = "/Users/yeeland/MLdata/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv(path, sep=',', names=names)
# print(df.head(20))

trainDF = pd.DataFrame()
# trainDF['data']
X = df.iloc[:, 0:4]

# trainDF['label']
Y = df.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

model = svm.SVC(gamma='scale').fit(x_train, y_train)

y_predict = model.predict(x_test)

accuracy = accuracy_score(y_predict, y_test)
print(accuracy)
