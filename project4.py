import os
from PIL import Image
import pandas
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np


# 加载数据集
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
                labels = np.append(labels, dir_name)
    return images, labels


image, label = get_data()
print(image.shape)

# 选择SVC模型
clf = SVC(kernel="linear")
# 训练
clf.fit(image, label)
# 预测
y_predict = clf.predict(image)
# 识别准确率
print(accuracy_score(label, y_predict))
