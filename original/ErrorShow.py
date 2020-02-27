# -*- coding=utf-8 -*-
import math

import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mpl


# 画出测试数据集的预测数据和测试数据
from sklearn.metrics import mean_squared_error


def draw(predData, trueData, lr, layers, sigmod):
    fig = plt.figure(figsize=(16, 8))
    for i in range(6):
        # 绘制子图，2行4列，表示8个属性的预测值与实际值信息
        rmse = math.sqrt(mean_squared_error(predData[:, i], trueData[:, i]))
        plt.subplot(int('24' + str(i+1)))
        plt.title('The ' + str(i) + '-th column attribute (rmse : %f):' % rmse)
        label_list = ['Actual value', 'Predictive value']
        plt.xlabel('Sample number')
        plt.ylabel('Attribute value')
        plt.plot(range(len(trueData)), trueData[:, i], c='r')
        plt.plot(range(len(predData)), predData[:, i], c='b')
        plt.legend(label_list, loc='best')
    fig.tight_layout()
    plt.savefig('fig/result(lr=%.2f layers=%d, sigmod=%.3f).png' % (lr, layers, sigmod))  # 将图片保存到文件
    plt.show()
