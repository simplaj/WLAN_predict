
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from pandas import *
import math

class clus:
    def __init__(self, clu=None):
        self.clu = clu

    def computer_max_distance(self):
        x = 0
        y = 0
        clu = np.array(self.clu)
        if clu.shape[0] == 0:
            print("None")
            return 0
        for i in range(clu.shape[0]):
            x = x + clu[i, 0]
            y = y + clu[i, 1]
        x_mean = x / clu.shape[0]
        y_mean = y / clu.shape[0]
        max = 0
        for i in range(clu.shape[0]):
            temp = math.sqrt((clu[i, 0] - x_mean) ** 2 + (clu[i, 1] - y_mean) ** 2)
            if max < temp:
                max = temp
        self.max_dis = max

    def best_connect_function(self):
        count = 0
        clu = np.array(self.clu)
        for i in range(clu.shape[1]):
            if clu[0, i] == -1:
                count = count + 1
        self.best_connect = count

    def clus_s(self,sum):
        self.clu_s = self.max_dis / sum * 7

    def computer_score(self, s_min, s_max):
        clu = np.array(self.clu)
        x = 0
        y = 0
        score = []
        s_min = s_min-self.clu_s
        for i in range(clu.shape[0]):
            x = x + clu[i, 0]
            y = y + clu[i, 1]
        x_mean = x / clu.shape[0]
        y_mean = y / clu.shape[0]
        for i in range(clu.shape[0]):
            temp = math.sqrt((clu[i, 0] - x_mean) ** 2 + (clu[i, 1] - y_mean) ** 2)
            if temp == 0:
                score.append(s_max)
            else:
                score.append((1 - temp / self.max_dis) * (s_max - s_min) + s_min)
        score = np.array(score).reshape(-1, 1)
        self.clu_score = np.append(clu, score, axis=-1)
'''def clustering_function(name):
    dataframe = read_csv(name+'.csv')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    x_test_encoded = dataset
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=0.007, min_samples=5).fit(x_test_encoded)
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=db.labels_, s=5)
    labels = np.array(db.labels_).reshape(-1,1)
    output = np.append(dataset,labels,axis=-1)
    save = DataFrame(output)
    save.columns = ['x', 'y', 'SP', 'RP', 'TI', 'TT', 'UT', 'LMS', 'CML', 'MLR', 'Label']
    save.to_csv('name_labels'+'.csv', index=False, header=True)
    plt.show()
    return output'''


def score_function(name):
    dataframe = read_csv(name + '.csv')
    dataframe1 = read_csv('data/normalization.csv')
    dataframe1.columns = ['n_SP', 'n_RP', 'n_TI', 'n_TT', 'n_UT', 'n_LMS', 'n_CML', 'n_MLR']
    dataframe = dataframe.join(dataframe1)
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    clus_count = 0
    only_one_clus_count = [0,0,0,0,0,0,0,0]
    for i in range(dataset.shape[0]):
        only_one_clus_count[int(dataset[i,10])] = only_one_clus_count[int(dataset[i,10])] +1   #计每种类别的数量
    for i in range(len(only_one_clus_count)):
        if only_one_clus_count[i] != 0:
            if only_one_clus_count[i]<=2:
                dataset[:,10] = dataset[:, 10]-1              #?
            else:
                clus_count = clus_count +1
    clu = []
    for i in range(int(clus_count)):
        _clus = clus(clu=[])
        clu.append(_clus)
    for i in range(dataset.shape[0]):
        for j in range(int(clus_count)):
            if dataset[i, 10] == j:
                clu[j].clu.append(dataset[i, :])
                break
    for i in range(len(clu)):
        clu[i].computer_max_distance()
        clu[i].best_connect_function()
    clu.sort(key=lambda x: x.best_connect, reverse=False)
    sum_distance = 0
    for i in range(len(clu)):
        sum_distance = sum_distance + clu[i].max_dis
    for i in range(len(clu)):
        clu[i].clus_s(sum_distance)
    last = 0
    for i in range(len(clu)):
        clu[i].computer_score(7-last,7-last)
        clu[i].clu_score[:,11] = clu[i].clu_score[:,11]
        last = clu[i].clu_s + last
    score = clu[0].clu_score
    for i in range(1,len(clu)):
        score = np.append(score, clu[i].clu_score, axis=0)

    save = DataFrame(score)
    label = save[10]
    score1 = save[19]
    save = save.drop(10, axis=1)
    save = save.drop(19, axis=1)
    save = save.join(label)
    save = save.join(score1)
    save.columns = ['x', 'y', 'SP', 'RP', 'TI', 'TT', 'UT', 'LMS', 'CML', 'MLR','n_SP', 'n_RP', 'n_TI', 'n_TT', 'n_UT', 'n_LMS', 'n_CML', 'n_MLR', 'Label','Score']
    save.to_csv('score_new.csv', index=False, header=True)


score_function('scores/outputs_new')
