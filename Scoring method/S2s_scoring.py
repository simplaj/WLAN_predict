import matplotlib.pyplot as plt
from pandas import DataFrame

from function import combine
from function import five_show
from function import LoadStruct
from function import ErrorShow
from function import LstmStruct
from pandas import DataFrame
sigmod = 0.01
timeSteps = 5  # 往前回溯的长度
train_ratio = 0.9  # 训练集所占的百分比
lstm_layers = 2  # LSTM隐藏层层数
lstm_hidden_size = 1  # LSTM每一层隐藏层节点数
epochs = 20000  # 训练周
lr = 0.08  # 学习速率


ld = LoadStruct.LoadData(train_ratio, timeSteps, sigmod)
encoder = LstmStruct.LstmData(ld, lstm_layers, lstm_hidden_size, timeSteps, epochs, lr)
en_Pred, en_True,  en_score, loss, pre_scores = encoder.lstmWork()  # 返回预测值和实际值
pre = en_Pred.reshape(en_Pred.shape[0], 70)
true = en_True.reshape(en_True.shape[0], 70)
ErrorShow.draw(pre[pre.shape[0]-2000:, :], true[true.shape[0]-2000:, :], '../pre/S2s_lstm_s2')  # 调用绘制结果图片
# print(en_states)
# print(en_score)
plt.plot(loss)
plt.show()
save = DataFrame(en_score.reshape([en_score.shape[0], en_score.shape[1]*en_score.shape[2]]))
save.to_csv('../scores/score_lstm_s2.csv', header=True, index=False)
save2 = DataFrame(pre_scores.reshape([pre_scores.shape[0], pre_scores.shape[1]*pre_scores.shape[2]]))
save2.to_csv('../pre/score_lstm_s2.csv', header=True, index=False)

# combine.combine('../scores/score_lstm_s.csv', '../pre/score_lstm_s.csv', '../scores/scores_lstm_s.csv')
# five_show.draw('../scores/scores_lstm_S.csv','../fig/scores_lstm_s.csv')
