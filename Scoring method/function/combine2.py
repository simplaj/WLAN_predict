import pandas as pd
import numpy as np
from sklearn import preprocessing
from pandas import DataFrame
import matplotlib.pyplot as plt

# Vae = pd.read_csv("../../scores/score_vae_s.csv")
Dae = pd.read_csv("../../scores/score_dae_s2.csv")
trans = pd.read_csv('../../scores/score_trans_s2.csv')
pretrans = pd.read_csv('../../pre/score_trans_s2.csv')
S2s = pd.read_csv('../../scores/score_lstm_s2.csv')
preS2s = pd.read_csv('../../pre/score_lstm_s2.csv')
Mean = pd.read_csv("../../scores/score_mean_s2.csv")
s2s = S2s.values[1:, 0]
pres2s = preS2s.values[1:, 0]
_trans = trans.values[1:, 0]
_pretrans = pretrans.values[1:, 0]
# vae = Vae.values[:s2s.shape[0], :]
dae = Dae.values[:s2s.shape[0], :]
mea = Mean.values[:s2s.shape[0], :]
output = np.append(mea.reshape(s2s.shape[0], 1), _trans.reshape(s2s.shape[0], 1), axis=-1)
output = np.append(output, _pretrans.reshape(s2s.shape[0], 1), axis=-1)
output = np.append(output, s2s.reshape(s2s.shape[0], 1), axis=-1)
output = np.append(output, pres2s.reshape(pres2s.shape[0], 1), axis=-1)
output = np.append(output, dae, axis=-1)

min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(output)

# data = np.append(C_s, data, axis=-1)
# data_ = data.values
save = DataFrame(data)
save.columns = ['mean', 'trans', 'pre_trans', 's2s', 'pres2s', 'dae']
save.to_csv('../../scores/scores_s911.csv', index=False, header=True)
