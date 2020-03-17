from keras import Model, Sequential
from keras.layers import Dense, LSTM, Input
from keras.optimizers import *
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# parameters for LSTM
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

nb_lstm_outputs = 30  # 神经元个数
timeSteps = 5  # 时间序列长度
d_input = 6  # 输入序列
train_ratio = 0.9
bts = 1145
epochs = 5000

input_Data = pd.read_csv('data/data0.csv').values
srcData = input_Data[:, 1:]
min_max_scaler = preprocessing.MinMaxScaler()
srcData = min_max_scaler.fit_transform(srcData)
len_data = srcData.shape[0]
datax = []
datay = []
for i in range(len(srcData) - timeSteps):
    datax.append(srcData[i: i + timeSteps])
    datay.append(srcData[i + timeSteps])
datax = np.array(datax)
datay = np.array(datay)
trainSet_len = int(len(datax) * train_ratio)
trainx, trainy, testx, testy = datax[:trainSet_len], datay[:trainSet_len], datax[trainSet_len:], datay[trainSet_len:]

input1 = Input(shape=(timeSteps, d_input))
# hiddens = LSTM(6, input_shape=(timeSteps, d_input), return_sequences=True)(input1)
# hiddens = LSTM(32, return_sequences=True)(hiddens)
hiddens = LSTM(6)(input1)
# output1 = Dense(32, activation='softmax')(hiddens)
output = Dense(6, activation='softmax')(hiddens)
model = Model(input1, output)

model.compile(optimizer='adam', loss='mean_absolute_error',
              metrics=['mse', 'acc'])
model.summary()
model.fit(trainx, trainy,
          batch_size=bts,
          epochs=epochs,
          validation_split=0.1
          )

pre = model.predict(testx)

fig = plt.figure(figsize=(16, 8))
for i in range(6):
    # 绘制子图，2行4列，表示8个属性的预测值与实际值信息
    rmse = np.math.sqrt(mean_squared_error(pre[:, i], testy[:, i]))
    plt.subplot(int('23' + str(i + 1)))
    plt.title('The ' + str(i) + '-th column attribute (rmse : %f):' % rmse)
    label_list = ['Actual value', 'Predictive value']
    plt.xlabel('Sample number')
    plt.ylabel('Attribute value')
    plt.plot(range(testy.shape[0]), testy[:, i], c='r')
    plt.plot(range(pre.shape[0]), pre[:, i], c='b')
    plt.legend(label_list, loc='best')
fig.tight_layout()
plt.savefig('fig/oters.png')  # 将图片保存到文件
plt.show()
