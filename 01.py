
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing

output = pd.read_csv('original\ss.csv')
output = output.values
output = output[1:, :]
min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(output)
df = DataFrame(data)
df.to_csv('outs.csv', index=True, header=False)