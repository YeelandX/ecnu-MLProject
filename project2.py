# -*- coding: UTF-8 -*-
import os
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas

# xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

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
            line = fh.read()
            try:
                line.decode("utf8")
            except:
                continue
                # print(file_path)
            texts.append(line)
            labels.append(dir_name)
            fh.close()

trainDF['text'] = texts
trainDF['label'] = labels
# 随机抽样1000行
trainDF = trainDF.sample(n=1000)
# 打乱数据
trainDF = trainDF.reset_index(drop=True)
# print(trainDF)

# 为标签分配连续编号
encoder = preprocessing.LabelEncoder()
trainDF['label'] = encoder.fit_transform(trainDF['label'])

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, test_y)


# 5重检验，划分训练集和测试集
kf = model_selection.KFold(n_splits=5)
for train_index, test_index in kf.split(trainDF['text']):
    train_x, test_x = trainDF['text'][train_index], trainDF['text'][test_index]
    train_y, test_y = trainDF['label'][train_index], trainDF['label'][test_index]

    # 特征工程：
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(trainDF['text'])
    xtrain_tfidf = tfidf_vect.transform(train_x)
    xtest_tfidf = tfidf_vect.transform(test_x)

    # 训练并预测
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xtest_tfidf)
    print("NB, WordLevel TF-IDF: ", accuracy)
