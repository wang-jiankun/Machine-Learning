"""
机器学习实战--朴素贝叶斯--demo
date：2018-7-13
author: 王建坤
"""
import numpy as np


# 创建数据集
def loadDataSet():
    # 每个样本为一段文字
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1表示不好，0表示好
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 从数据集中生成词汇表
def createVocabList(dataSet):
    # 创建空集合
    vocabSet = set([])
    # 提取数据集中所有的词汇，不重复
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 把文本转化为特征向量
def setOfWords2Vec(vocabList, inputSet):
    # 创建特征向量，全0
    returnVec = [0]*len(vocabList)
    # 文本中出现的词汇相应的在特征向量中对应的位置置1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


# 创建贝叶斯分类器，输入用特征向量表示的数据集及数据集的标记
def trainNB(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 计算数据集中不好语句的占比
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 存放不同类别中各词汇出现的次数
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    # 存放不同类别中词汇的总数
    p0Denom = 2.0
    p1Denom = 2.0
    # 遍历数据集，计算上述四个值
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 计算不同类别中各词汇出现的概率
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive


# 用朴素的贝叶斯分类
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 计算该样本属于不同类别的概率，注意加了一个log
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    # 根据概率预测结果
    if p1 > p0:
        return 1
    else:
        return 0


# 把数据集转换为用特征向量表示样本的数据集
def getTrainMat(trainSet, myVocabList):
    trainMat = []
    for example in trainSet:
        trainMat.append(setOfWords2Vec(myVocabList, example))
    return trainMat


if __name__ == '__main__':
    # 加载数据集
    trainSet, labels = loadDataSet()
    # 构建词汇表
    myVocabList = createVocabList(trainSet)
    # 得到用特征向量表示样本的数据集
    trainMat = getTrainMat(trainSet, myVocabList)
    # 朴素的贝叶斯分类器得到模型参数
    p0V, p1V, pAb = trainNB(trainMat, labels)
    # print(p0V, '\n', p1V, '\n', pAb)
    # 测试分类器
    test = ['cute', 'has']
    testVec = np.array(setOfWords2Vec(myVocabList, test))
    predict = classifyNB(testVec, p0V, p1V, pAb)
    print(test, 'be predicted as ', predict)
