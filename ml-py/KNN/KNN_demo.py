"""
机器学习实战--KNN算法--demo
date：2018-7-12
author: 王建坤
"""
import numpy as np


# 创建数据集
def createDataset():
    data_set = np.array([[1., 1.1], [1., 1.], [0., 0.], [0., 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return data_set, labels


# 创建分类器
def classify(x, data_set, labels, k):
    # 计算欧式距离
    distances = (((data_set - x)**2).sum(axis=1))**0.5
    # 对距离进行排序，返回排序的下标
    sort_distances = distances.argsort()

    # 存放前k个邻近样本属于各种类别的个数的字典
    class_count = {}
    for i in range(k):
        # 第i个样本的类名
        class_name = labels[sort_distances[i]]
        # 存放在字典中，如果类名相同值加1
        class_count[class_name] = class_count.get(labels[sort_distances[i]], 0) + 1
    # 对类别的个数进行排序
    sort_class_count = sorted(class_count.items(), key=lambda dic: dic[1], reverse=True)
    # 返回类别个数最多的类名
    return sort_class_count[0][0]


if __name__ == '__main__':
    x1 = [1.1, 1.1]
    data_set1, labels1 = createDataset()
    result = classify(x1, data_set1, labels1, 3)
    print(x1, 'is: ', result)
