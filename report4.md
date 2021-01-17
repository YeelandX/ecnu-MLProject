## 实验报告4：人脸识别
### 实验目标
CMU Machine Learning Faces数据集：
http://www.cs.cmu.edu/afs/cs.cmu.edu/user/mitchell/ftp/faces.html

任务1：使用机器学习进行人脸分类识别，给出识别准确率

任务2：使用聚类或分类算法发现表情相似的脸图

### 实验思路
ML算法：支持向量机

### 实验过程

1. 获取数据集。下载数据集合至本地，遍历目录获取数据集和标签。

        def get_data():
        images = np.ndarray((1, 28 * 28))
        labels = np.ndarray(1)
        path = "/Users/yeeland/MLdata/faces_4"
        for dir_name in os.listdir(path):
            dir_path = os.path.join(path, dir_name)
            try:
                os.listdir(dir_path)
            except BaseException:
                continue
            else:
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    img = Image.open(file_path)
                    img = img.resize((28, 28))
                    images = np.concatenate((images, np.array(img).reshape(1, -1)), axis=0)
                    # 任务一：
                    labels = np.append(labels, dir_name)
                    # 任务二：提取文件名中的表情关键字作为标签
                    labels = np.append(labels, (file.split('_'))[2])
        return images, labels

        # 获取数据集
        image, label = get_data()
        print(image.shape)
2. 选择sklearn.svm包中SVC()方法训练模型。
    
        # 选择SVC模型
        clf = SVC(kernel="linear")
        # 训练
        clf.fit(image, label)
        
3. 预测并输出模型准确率
    
        # 预测
        y_predict = clf.predict(image)
        # 识别准确率
        print('accuracy:', accuracy_score(label, y_predict))

### 实验结果

任务1：
![avatar](/Users/yeeland/MLdata/pro4_1.png)

如图所示，运行得到的训练准确率约为0.985。


任务2：
![avatar](/Users/yeeland/MLdata/pro4_2.png)

如图所示，运行得到的训练准确率约为0.933。