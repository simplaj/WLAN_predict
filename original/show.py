import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

orig_score = pd.read_csv("scores/score_2.csv")
data = orig_score.values
data1 = data[:, 21]
data2 = data[:, 22]
fig = plt.figure(figsize=(16, 8))

plt.plot(data1)
# plt.plot(data2, label="vae")
plt.legend(loc='best')
plt.show()

