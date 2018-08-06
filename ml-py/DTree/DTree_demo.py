"""
机器学习实战--决策树（decision tree）--demo
date：2018-7-13
author: 王建坤
"""
import numpy as np
import math


# 创建数据集
def createSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    # 特征属性的标记
    labels = ['no sufacing', 'flippers']
    return dataSet, labels


# 计算数据集的信息熵
def calcEntropy(dataset):
    numEntries = len(dataset)
    # 存放各个类别的样本个数
    labelCounts = {}
    for featVect in dataset:
        currentLable = featVect[-1]
        if currentLable not in labelCounts.keys():
            labelCounts[currentLable] = 0
        labelCounts[currentLable] += 1
    shannonEnt = 0.0
    # 计算信息熵
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt


# 根据特征的特征值找到这类样本组成数据集
def splitDataset(dataSet, feature, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[feature] == value:
            reduceFeatVec = featVec[:feature]
            reduceFeatVec.extend(featVec[feature+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet


# 选择最优的特征
def chooseBestFeature(dataSet):
    numFeature = len(dataSet[0]) - 1
    baseEntropy = calcEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeature):
        featValue = [x[i] for x in dataSet]
        featSet = set(featValue)
        newEntropy = 0.0
        # 计算按照该特征划分的信息熵
        for value in featSet:
            subDataSet = splitDataset(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcEntropy(subDataSet)
        # 计算按照该特征划分的信息熵增益
        infoGain = baseEntropy - newEntropy
        # print(infoGain)
        # 如果信息增益比之前的信息增益大，更新最好信息增益的值，并记录该特征为新的最好特征
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 选择类别数目最多的类
def chooseClass(classList):
    classCount = {}
    # 计算各个类别的数目
    for c in classList:
        if c not in classCount.keys():
            classCount[c] = 0
        classCount[c] += 1
    # 对数目进行排序，降序
    sortClassCount = sorted(classCount.items(), key=lambda dic: dic[1], reverse=True)
    # 返回数目最大的类别
    return sortClassCount[0][0]


# 递归创建树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 如果数据集的类别都一样，返回该类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 当所有的特征都被遍历了，返回数据集中最多的类别
    if len(dataSet[0]) == 1:
        return chooseClass(classList)
    # 选择最优的特征
    bestFeat = chooseBestFeature(dataSet)
    bestFeatLabel = labels[bestFeat]
    # 把该特征作为节点存放在字典中
    myTree = {bestFeatLabel: {}}
    # 把该特征从特征向量中剔除
    del (labels[bestFeat])
    # 对于该特征划分生成的子集进行递归划分
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 复制特征标记
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataset(dataSet, bestFeat, value), subLabels)
    return myTree


# 用决策树进行分类
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


if __name__ == '__main__':
    # 创建树
    dataSet, labels = createSet()
    tree = createTree(dataSet, labels)
    print(tree)
    # 测试树
    dataSet, labels = createSet()
    test = [1, 1]
    predict = classify(tree, labels, test)
    print(test, "be predicted as: ", predict)
