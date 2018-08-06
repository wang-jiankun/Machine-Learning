"""
机器学习实战--KNN算法--约会
date：2018-7-12
author: 王建坤
"""
import numpy as np
from matplotlib import pyplot as plt
from KNN.KNN_demo import classify


# 数据集文件转为矩阵
def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    numberOfLines = len(lines)
    dataset = np.zeros([numberOfLines, 3])
    labels = []

    for index, line in enumerate(lines):
        line = line.strip()
        listFromLine = line.split('\t')
        dataset[index, :] = listFromLine[0:3]
        labels.append(int(listFromLine[-1]))

    return dataset, labels


# 数据集图形化
def draw(x, y):
    plt.scatter(x[:, 1], x[:, 2], 15*labels, 30*np.array(labels))
    plt.show()


# 数据归一化
def dataNorm(dataset):
    minVals, maxVals = dataset.min(0), dataset.max(0)
    ranges = maxVals - minVals
    dataset = (dataset - minVals)/ranges
    return dataset


# 测试分类器，计算准确率
def calcAccuracy(datasetNorm, labels):
    testRatio = 0.1
    testNum = int(testRatio*datasetNorm.shape[0])
    print(testNum)
    errorCount = 0
    for i in range(testNum):
        predict = classify(datasetNorm[i, :], datasetNorm[testNum:datasetNorm.shape[0], :],
                           labels[testNum:datasetNorm.shape[0]], 3)
        print('predict:', predict, ' true:', labels[i])
        if predict != labels[i]:
            errorCount += 1
    print('accuracy is: ', (testNum-errorCount)/testNum)


if __name__ == '__main__':
    dataset, labels = file2matrix('datingTestSet2.txt')
    print(dataset, labels)
    draw(dataset, labels)
    datasetNorm = dataNorm(dataset)
    print(datasetNorm)
    calcAccuracy(datasetNorm, labels)


