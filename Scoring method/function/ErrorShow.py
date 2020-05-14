# -*- coding=utf-8 -*-
import math

import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mpl


# 画出测试数据集的预测数据和测试数据
from sklearn.metrics import mean_squared_error


def draw(predData, trueData, name):
    i = 0
    fig = plt.figure(figsize=(24, 12))
    for name1 in ['RP', 'TI', 'TT', 'UT', 'LMS', 'CML', 'MLR', 'RP2', 'TI2', 'TT2', 'UT2', 'LMS2', 'CML2', 'MLR2']:
        # 绘制子图，3行4列，表示12个属性的预测值与实际值信息
        rmse = math.sqrt(mean_squared_error(predData[:, i], trueData[:, i]))
        plt.subplot(2, 7, i+1)
        plt.title(name1 + ' (rmse : %f):' % rmse)
        label_list = ['Actual value', 'Predictive value']
        plt.xlabel('Sample number')
        plt.ylabel('Attribute value')
        plt.plot(range(trueData.shape[0]), trueData[:, i], c='r')
        plt.plot(range(predData.shape[0]), predData[:, i], c='b')
        plt.legend(label_list, loc='best')
        i += 1
    fig.tight_layout()
    plt.savefig(name+'.png' )  # 将图片保存到文件
    plt.show()
