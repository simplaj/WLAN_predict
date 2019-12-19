import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import DataFrame

data = pd.read_csv("scores/score_1.csv")
dae_ = pd.read_csv("scores/score_mean.csv")
dae = dae_.values
data['mean'] = dae[:, 0]
data_ = data.values
save = DataFrame(data_)
save.columns = ['x', 'y', 'SP', 'RP', 'TI', 'TT', 'UT', 'LMS', 'CML', 'MLR', 'n_SP', 'n_RP', 'n_TI',
                'n_TT', 'n_UT', 'n_LMS', 'n_CML', 'n_MLR', 'Label',
                'clu&sco', 'mean', 'c&s_0_1', 'vae', 's2s', 'dae']
save = save.sort_values(by='SP', ascending=True)
save.to_csv('score_1.csv', index=False, header=True)
