import matplotlib.pyplot as plt
import LstmStruct
import LoadStruct
import ErrorShow
from pandas import DataFrame

sigmod = 0.01
timeSteps = 5  # 往前回溯的长度
train_ratio = 0.9  # 训练集所占的百分比
lstm_layers = 2  # LSTM隐藏层层数
lstm_hidden_size = 1  # LSTM每一层隐藏层节点数
epochs = 2000  # 训练周期
lr = 0.08  # 学习速率
ld = LoadStruct.LoadData(train_ratio, timeSteps, sigmod)
encoder = LstmStruct.LstmData(ld, lstm_layers, lstm_hidden_size, timeSteps, epochs, lr)
en_Pred, en_True,  en_score, loss = encoder.lstmWork()  # 返回预测值和实际值
# decoder = LstmStruct.LstmData(ld, lstm_layers, lstm_hidden_size, timeSteps, epochs, lr)

# decoder = LstmStruct.LstmData(ld, lstm_layers, lstm_hidden_size, timeSteps, epochs, lr)
# testPred, testTrue, en_states, score = encoder.lstmWork()  # 返回预测值和实际值

ErrorShow.draw(en_Pred, en_True, lr, lstm_layers, sigmod)  # 调用绘制结果图片
# print(en_states)
# print(en_score)
plt.plot(loss)
plt.show()
save = DataFrame(en_score.reshape([en_score.shape[0], en_score.shape[1]*en_score.shape[2]]))
save.to_csv('score_t.csv', header=True, index=False)
