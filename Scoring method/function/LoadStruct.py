# -*- coding=utf-8 -*-

import pandas as pd
import numpy as np
import random


class LoadData():
    def __init__(self, train_ratio, timeSteps, sigmod):
        srcDta = pd.read_csv('../data/simdata.csv').values  # 读入数据
        srcDta = srcDta[:, 1:]
        # 将源数据格式转换为LSTM网络要求的格式，其中inputData是LSTM的输入数据，outputData是实际的预测数据
        inputData, outputData = self.createData(srcDta, timeSteps)
        # 划分楚训练集和测试集
        self.trainx, self.trainy, self.testx, self.testy = self.splitData(inputData, outputData, train_ratio)
        self.trainx_n = self.addNoise(self.trainx)
        self.testx_n = self.addNoise(self.testx)

    # 根据参数划分训练集和测试集，得到各自的输入数据和输出数据
    def splitData(self, inputData, outputData, train_ratio):
        trainSet_len = int(len(inputData) * train_ratio)
        return (inputData[:trainSet_len], outputData[:trainSet_len],
                inputData[trainSet_len:], outputData[trainSet_len:])

    def createData(self, srcData, timeSteps):
        datax = [];
        datay = []
        for i in range(len(srcData) - timeSteps):
            datax.append(srcData[i: i + timeSteps])
            datay.append(srcData[i + 1: i + timeSteps + 1])
        datax = np.array(datax)
        datay = np.array(datay)
        return datax, datay

    def addNoise(self, data):
        noise = np.random.normal(loc=0.5, scale=0.5, size=data.shape)
        x_n = data + noise

        return x_n
