import pandas as pd
import numpy as np
from sklearn import preprocessing
from pandas import DataFrame
import matplotlib.pyplot as plt

Vae = pd.read_csv("scores/VAE_score.csv")
Dae = pd.read_csv("scores/score_dae_1.csv")
Mea = pd.read_csv("scores/score_mean.csv")
S2s = pd.read_csv("scores/s2s_score_n_512.csv")
C_s = pd.read_csv("scores/score_new.csv")
C_s = C_s.sort_values(by='SP', ascending=True)
c_s = C_s['Score']
vae = Vae.values[1:, :]
dae = Dae.values[1:, :]
mea = Mea.values[1:, :]
s2s = S2s.values[1:, 4]
c_s = c_s.values
output = np.append(mea, c_s.reshape(C_s.shape[0], 1), axis=-1)
output = np.append(output, vae, axis=-1)
output = np.append(output, s2s.reshape(C_s.shape[0], 1), axis=-1)
output = np.append(output, dae, axis=-1)

min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(output)

data = np.append(C_s, data, axis=-1)
# data_ = data.values
save = DataFrame(data)
save.columns = ['x', 'y', 'SP', 'RP', 'TI', 'TT', 'UT', 'LMS', 'CML', 'MLR', 'n_SP',
                'n_RP', 'n_TI', 'n_TT', 'n_UT', 'n_LMS', 'n_CML', 'n_MLR', 'Label',
                'clu&sco', 'mean', 'c&s_0_1', 'vae', 's2s', 'dae']
save.to_csv('score_2.csv', index=False, header=True)
