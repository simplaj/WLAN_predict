import math

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

Tri_data = pd.read_csv('../../scores/scores_aaa.csv')
tri_data = Tri_data.values
begin = 0
length = int(tri_data.shape[0] / 3000)
for ii in range(int(length / 2) + 1):
    fig = plt.figure(figsize=(16, 8))
    for i in range(2):
        if begin < 3000 * length:
            end = begin + 3000
        else:
            end = len(tri_data)
        if begin == end:
            break
        mean_data1 = tri_data[begin:end, 0]
        trans_data1 = tri_data[begin:end, 1]
        pretrans_data1 = tri_data[begin:end, 2]
        # vae_data1 = tri_data[begin:end, 3]
        s2s_data1 = tri_data[begin:end, 3]
        pres2s_data1 = tri_data[begin:end, 4]
        dae_data1 = tri_data[begin:end, 5]

        rmse1 = math.sqrt(mean_squared_error(tri_data[begin:end, 1], tri_data[begin:end, 2]))
        rmse2 = math.sqrt(mean_squared_error(tri_data[begin:end, 4], tri_data[begin:end, 5]))

        plt.subplot(211 + i)
        plt.title("Scores(%d-%d lines rmse=%f )" % (begin, end,rmse2))
        # plt.plot(trans_data1, label='trans_score')
        # plt.plot(mean_data1, label='mean_score')
        plt.plot(s2s_data1, label='seq2seq')
        # plt.plot(dae_data1, label='dae_score')
        # plt.plot(vae_data1, label='vae_score')
        plt.plot(pres2s_data1, label='pres2s')
        # plt.plot(pretrans_data1, label='pretrans')
        plt.legend(loc='best')
        # plt.show()
        begin = end

    fig.savefig('../../fig/apl_%d.png' % ii)
    plt.show()
