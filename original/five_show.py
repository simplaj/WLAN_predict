import pandas as pd
import matplotlib.pyplot as plt

Tri_data = pd.read_csv("score_3.csv")
tri_data = Tri_data.values
for ii in range(4):
    begin = 6000*ii
    end = begin+1000
    fig = plt.figure(figsize=(24, 16))
    for i in range(6):
        dae_data1 = tri_data[begin:end, 24]
        mean_data1 = tri_data[begin:end, 20]
        vae_data1 = tri_data[begin:end, 22]
        ori_data1 = tri_data[begin:end, 21]
        s2s_data1 = tri_data[begin:end, 23]

        plt.subplot(231+i)
        plt.title("Scores(%d-%d lines)" % (begin, end))
        plt.plot(dae_data1, label='dae_score')
        plt.plot(mean_data1, '-x', label='mean_score')
        plt.plot(s2s_data1, label='seq2seq')
        plt.plot(ori_data1, label='clustering&scoring')
        plt.plot(vae_data1, label='vae_score')
        plt.legend(loc='best')
        # plt.show()
        if begin <= 18000:
            begin += 1000
            end += 1000
        elif begin == 19000:
            end = 19070
    fig.savefig('fig/score_att_%d.png' % ii)
    plt.show()
