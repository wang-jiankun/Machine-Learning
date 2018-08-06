"""
机器学习实战--KNN算法--手写数字识别
date：2018-7-12
author: 王建坤
"""
import numpy as np
import os
from KNN.KNN_demo import classify


def img2vector(filename):
    vector = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            vector[0, 32*i+j] = int(lineStr[j])
    return vector


def handwritingClassTest():
    labels = []
    trainFileList = os.listdir('trainingDigits')
    trainNum = len(trainFileList)
    trainMat = np.zeros((trainNum, 1024))
    for i in range(trainNum):
        fileNameStr = trainFileList[i]
        classNum = fileNameStr.split('.')[0].split('_')[0]
        labels.append(classNum)
        trainMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('testDigits')
    errorCount = 0
    testNum = len(testFileList)
    for i in range(testNum):
        fileNameStr = testFileList[i]
        classNum = fileNameStr.split('.')[0].split('_')[0]
        testVector = img2vector('testDigits/%s' % fileNameStr)
        predict = classify(testVector, trainMat, labels, 3)
        print('predict: ', predict, 'true: ', classNum)
        if predict != classNum:
            errorCount += 1
    print('accuracy is: ', (testNum - errorCount) / testNum)


if __name__ == '__main__':
    handwritingClassTest()