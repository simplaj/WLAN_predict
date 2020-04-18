import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

dataFrame = pd.read_csv('data.csv').values
st = str(dataFrame[0, 0])
sp = str(dataFrame[0, 1])
send = [st.split(' '), sp.split(' ')]
sd = DataFrame(send, dtype=np.float)
sdt = sd.T
sdt.columns = ['st', 'p']
sdt['sp'] = '0'
sdt['ti'] = '0'
for i in range(sdt.shape[0]):
    if i == 0:
        sdt['sp'][0] = sdt['p'][0]
        sdt['ti'][0] = -1
    else:
        sdt['sp'][i] = sdt['sp'][i - 1] + sdt['p'][i]
        sdt['ti'][i] = sdt['st'][i] - sdt['st'][i-1]

sdt.to_csv('send.csv', header=True, index=False)


rt = str(dataFrame[0, 2])
rp = str(dataFrame[0, 3])
recv = [rt.split(' '), rp.split(' ')]
re = DataFrame(recv, dtype=np.float)
ret = re.T
ret.columns = ['rt', 'p']
ret['rp'] = '0'
ret['ut'] = '0'
for i in range(ret.shape[0]):
    if i == 0:
        ret['rp'][0] = ret['p'][0]
        ret['ut'][0] = -1
    else:
        ret['rp'][i] = ret['rp'][i - 1] + ret['p'][i]
        ret['ut'][i] = ret['rt'][i] - ret['rt'][i-1]

ret.to_csv('recv.csv', header=True, index=False)
data = sdt

data['SP'] = '0'
data['RP'] = '0'
data['TI'] = '0'
data['TT'] = '0'
data['UT'] = '0'
data['LMS'] = '0'
data['CML'] = '0'
data['MLR'] = '0'
flag = 0
lm = 0
for i in range(sdt.shape[0]-2):
    data['TT'][i] = ret['rt'][flag] - sdt['st'][i]
    data['SP'][i] = sdt['sp'][i]
    data['RP'][i] = ret['rp'][flag]
    data['UT'][i] = ret['ut'][flag]
    data['TI'][i] = sdt['ti'][i]
    if data['TT'][i] > 0.10001:
        flag = flag
        data['TT'][i] = -1
        data['UT'][i] = -1
    else:
        flag = flag+1
    print(i,flag)
    if data['TT'][i] == -1:
        lm = lm+1
        data['LMS'][i] = 1
        data['MLR'][i] = (lm+1)/data['SP'][i]
    else:
        data['LMS'][i] = 0
        data['MLR'][i] = lm/data['SP'][i]
    if i == 0:
        data['CML'][i] = data['LMS'][i]
    else:
        if (data['CML'][i-1] != 0) and (data['TT'][i] == -1):
            data['CML'][i] = int(data['CML'][i-1])+1
data.to_csv('output1.csv', header=True, index=False)