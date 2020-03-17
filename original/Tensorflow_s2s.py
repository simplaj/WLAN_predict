import matplotlib.pyplot as plt
import LstmStruct
import LoadStruct
import ErrorShow
from pandas import DataFrame
import combine
import five_show
import check
sigmod = 0.01
timeSteps = 5  # 往前回溯的长度
train_ratio = 0.9  # 训练集所占的百分比
lstm_layers = 2  # LSTM隐藏层层数
lstm_hidden_size = 1  # LSTM每一层隐藏层节点数
epochs = 20000  # 训练周
lr = 0.08  # 学习速率
s2s_name = 's2s_55.csv'
scores_name = 'scores_55.csv'
fig_name = 'scores_55'


# decoder = LstmStruct.LstmData(ld, lstm_layers, lstm_hidden_size, timeSteps, epochs, lr)

# decoder = LstmStruct.LstmData(ld, lstm_layers, lstm_hidden_size, timeSteps, epochs, lr)
# testPred, testTrue, en_states, score = encoder.lstmWork()  # 返回预测值和实际值

ld = LoadStruct.LoadData(train_ratio, timeSteps, sigmod)
encoder = LstmStruct.LstmData(ld, lstm_layers, lstm_hidden_size, timeSteps, epochs, lr)
en_Pred, en_True,  en_score, loss, pre_scores = encoder.lstmWork()  # 返回预测值和实际值
ErrorShow.draw(en_Pred.reshape(en_Pred.shape[0], 30), en_True.reshape(en_True.shape[0],30), 'no_neg55')  # 调用绘制结果图片
# print(en_states)
# print(en_score)
plt.plot(loss)
plt.show()
save = DataFrame(en_score.reshape([en_score.shape[0], en_score.shape[1]*en_score.shape[2]]))
save.to_csv(s2s_name, header=True, index=False)
save2 = DataFrame(pre_scores.reshape([pre_scores.shape[0], pre_scores.shape[1]*pre_scores.shape[2]]))
save2.to_csv('pre_'+s2s_name, header=True, index=False)

combine.combine(s2s_name, 'pre_'+s2s_name, scores_name)
five_show.draw(scores_name, fig_name)
for name in ['s2s', 'pres2s']:
    print(name+'checking...')
    check.check(name, scores_name)