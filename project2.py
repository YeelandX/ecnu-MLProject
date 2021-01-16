# -*- coding: UTF-8 -*-
import os
from sklearn import model_selection, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas
from sklearn.svm import SVC


def get_data():
    texts = []
    labels = []
    path = "/Users/yeeland/MLdata/20_newsgroups"
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
                except BaseException:
                    continue
                texts.append(line)
                labels.append(dir_name)
            fh.close()
    return texts, labels


if __name__ == '__main__':
    # 获取数据集
    trainDF = pandas.DataFrame()
    trainDF['text'], trainDF['label'] = get_data()
    # 随机抽样1000行
    trainDF = trainDF.sample(n=1000)
    # 打乱数据
    trainDF = trainDF.reset_index(drop=True)
    # 计算文本词语级TF-IDF值作为文本特征
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    text_tfidf = tfidf_vect.fit_transform(trainDF['text'])
    print(trainDF.shape)

    # 5重检验
    kf = model_selection.KFold(n_splits=5)
    for train_index, test_index in kf.split(trainDF['text']):
        # 划分训练集和测试集
        train_x, test_x = text_tfidf[train_index], text_tfidf[test_index]
        train_y, test_y = trainDF['label'][train_index], trainDF['label'][test_index]
        # 训练模型
        model = SVC()
        model.fit(train_x, train_y)
        # 预测
        predictions = model.predict(test_x)
        accuracy = metrics.accuracy_score(predictions, test_y)
        print("accuracy: ", accuracy)
