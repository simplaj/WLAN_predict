import pandas as pd
import numpy as np
from sklearn import preprocessing
from pandas import DataFrame
import matplotlib.pyplot as plt
def combine(name1, name2, outputname):
    Vae = pd.read_csv("scores/vae_scores.csv")
    Dae = pd.read_csv("scores/score_dae_1.csv")
    S2s = pd.read_csv(name1)
    preS2s = pd.read_csv(name2)
    C_s = pd.read_csv("scores/score_new.csv")
    C_s = C_s.sort_values(by='SP', ascending=True)
    c_s = C_s['Score']
    s2s = S2s.values[1:, 0]
    pres2s = preS2s.values[1:, 0]
    C_s = C_s.values[:s2s.shape[0], :]
    score = C_s[:, 12:17]
    vae = Vae.values[:s2s.shape[0], :]
    dae = Dae.values[:s2s.shape[0], :]
    mea = np.mean(score, 1)
    c_s = c_s.values[:s2s.shape[0]]
    output = np.append(mea.reshape(s2s.shape[0],1), c_s.reshape(s2s.shape[0], 1), axis=-1)
    output = np.append(output, vae, axis=-1)
    output = np.append(output, s2s.reshape(s2s.shape[0], 1), axis=-1)
    output = np.append(output, pres2s.reshape(pres2s.shape[0], 1), axis=-1)
    output = np.append(output, dae, axis=-1)

    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(output)

    data = np.append(C_s, data, axis=-1)
    # data_ = data.values
    save = DataFrame(data)
    save.columns = ['x', 'y', 'SP', 'RP', 'TI', 'TT', 'UT', 'LMS', 'CML', 'MLR', 'n_SP',
                    'n_RP', 'n_TI', 'n_TT', 'n_UT', 'n_LMS', 'n_CML', 'n_MLR', 'Label',
                    'clu&sco', 'mean', 'c&s_0_1', 'vae', 's2s', 'pres2s', 'dae']
    save.to_csv(outputname, index=False, header=True)
# combine('transformert.csv', 'pre_transformert.csv', 'tran_scoretv.csv')