## 实验报告2:文本分类
### 实验目标
数据集：http://www.cs.cmu.edu/afs/cs/project/theo-11/www/naive-bayes.html

任务：1000个文档分成20类，五重交叉验证结果

### 实验思路

ML算法：朴素贝叶斯

### 实验过程
1. 获取数据集。下载数据至本地，遍历每一个目录下的文件，将目录名作为标签。

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
2. 随机选取数据集中的1000行并打乱数据集。

        # 随机抽样1000行
        trainDF = trainDF.sample(n=1000)
        # 打乱数据
        trainDF = trainDF.reset_index(drop=True)

3. 特征工程。使用sklearn.feature_extraction.text包中的TfidfVectorizer()计算所有文本词语级的TF-IDF作为文本特征。

        # 计算文本词语级TF-IDF值作为文本特征
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
        text_tfidf = tfidf_vect.fit_transform(trainDF['text'])
        print(trainDF.shape)
4. 5重交叉检验。使用sklearn.model_selection包中KFold()方法将训练数据集分为5份。
每次选取其中一份作为测试集，另外四份作为训练集，迭代5次。

        kf = model_selection.KFold(n_splits=5)
        for train_index, test_index in kf.split(trainDF['text']):
        
5. 训练并预测。每次选取训练集和测试集，使用sklearn.naive_bayes包中MultinomialNB()方法训练模型。

        # 选取训练集和测试集
        train_x, test_x = text_tfidf[train_index], text_tfidf[test_index]
        train_y, test_y = trainDF['label'][train_index], trainDF['label'][test_index]
        # 训练模型
        model = naive_bayes.MultinomialNB()
        model.fit(train_x, train_y)
        # 预测
        predictions = model.predict(test_x)
        accuracy = metrics.accuracy_score(predictions, test_y)
        print("accuracy: ", accuracy)

### 实验结果

![avatar](/Users/yeeland/MLdata/pro2.png)

如图所示，五重交叉检验运行得到的训练准确率分别为0.53、0.48、0.525、0.555、0.66。