# -*- coding: UTF-8 -*-
import os
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas

# xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence

# from keras import layers, models, optimizers

# 加载数据集
trainDF = pandas.DataFrame()
texts = []
labels = []
path = "/Users/yeeland/20_newsgroups"
for dir_name in os.listdir(path):
    dir_path = os.path.join(path, dir_name)
    try:
        os.listdir(dir_path)
    except BaseException:
        continue
    else:
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            fh = open(file_path, "rb")
            texts.append(fh.read())
            labels.append(dir_name)
            fh.close()

trainDF['text'] = texts
trainDF['label'] = labels
# print(trainDF)
print(trainDF.shape)

# 划分数据集
train_x, test_x, train_y, test_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

# 特征工程ttt

