# 求平均值
import pandas as pd
import numpy as np
from pandas import DataFrame


Data = pd.read_csv("data/normalization.csv")
Data = Data.values
score = Data[:, 2:]
score = np.mean(score, 1)
data = DataFrame(score)
data.to_csv("score_mean.csv", index=False)
