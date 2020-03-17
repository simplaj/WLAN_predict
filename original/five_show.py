import math

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def draw(_in, _out):
    Tri_data = pd.read_csv(_in)
    tri_data = Tri_data.values
    begin = 0
    for ii in range(4):
        fig = plt.figure(figsize=(16, 8))
        for i in range(2):
            if begin < 18000:
                end = begin + 3000
            else:
                end = len(tri_data)
            if begin == end:
                break
            dae_data1 = tri_data[begin:end, 25]
            mean_data1 = tri_data[begin:end, 20]
            vae_data1 = tri_data[begin:end, 22]
            ori_data1 = tri_data[begin:end, 21]
            s2s_data1 = tri_data[begin:end, 23]
            pres2s_data1 = tri_data[begin:end, 24]

            # rmse = math.sqrt(mean_squared_error(tri_data[begin:end, 23], tri_data[begin:end, 24]))

            plt.subplot(211+i)
            plt.title("Scores(%d-%d lines )" % (begin, end))
            plt.plot(dae_data1, label='dae_score')
            plt.plot(mean_data1, label='mean_score')
            plt.plot(s2s_data1, label='seq2seq')
            plt.plot(ori_data1, label='clustering&scoring')
            plt.plot(vae_data1, label='vae_score')
            plt.plot(pres2s_data1, c='b', label='pres2s')
            plt.legend(loc='best')
            # plt.show()
            begin = end

        fig.savefig('fig/'+_out+'%d.png' % ii)
        plt.show()
# draw('tran_scoretv.csv', 'transformertv')