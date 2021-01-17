import os
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


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
                labels = np.append(labels, (file.split('_'))[2])
    return images, labels


if __name__ == '__main__':
    # 获取数据集
    image, label = get_data()
    print(image.shape)
    # 选择SVC模型
    clf = SVC(kernel="linear")
    # 训练
    clf.fit(image, label)
    # 预测
    y_predict = clf.predict(image)
    # 识别准确率
    print('accuracy:', accuracy_score(y_predict, label))
