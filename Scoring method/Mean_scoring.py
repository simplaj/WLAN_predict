# 求平均值
import pandas as pd
import numpy as np
from pandas import DataFrame
import time

Data = pd.read_csv("../data/simdata.csv")
Data = Data.values
score = Data[:, 1:]
mb = time.time()
score = np.mean(score, 1)
me = time.time()
print('score time = %f'%(me-mb))
data = DataFrame(score)
data.to_csv("../scores/score_mean_s2.csv", index=False)
