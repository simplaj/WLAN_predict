from keras.engine.saving import load_model
from pandas import *
from keras.layers import *
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

batch_size = 256
epochs = 40
encoding_dim = 2

dataframe = read_csv('data/addDim51.csv')
dataset = dataframe.values
dataset = dataset.astype('float32')
orignal_data = dataset

label = []

for i in range(orignal_data.shape[0]):
    if orignal_data[i,2] == -1 and orignal_data[i,3] == -1 and orignal_data[i,4] == -1:
        label.append(0)
    if orignal_data[i,2] == -1 and orignal_data[i,3] == -1 and orignal_data[i,4] != -1:
        label.append(1)
    if orignal_data[i,2] == -1 and orignal_data[i,3] != -1 and orignal_data[i,4] != -1:
        label.append(2)
    if orignal_data[i,2] != -1 and orignal_data[i,3] == -1 and orignal_data[i,4] == -1:
        label.append(3)
    if orignal_data[i,2] != -1 and orignal_data[i,3] == -1 and orignal_data[i,4] != -1:
        label.append(4)
    if orignal_data[i,2] != -1 and orignal_data[i,3] != -1 and orignal_data[i,4] != -1:
        label.append(5)
label = np.array(label)


dataframe = read_csv('data/normalization.csv')
dataset = dataframe.values
dataset = dataset.astype('float32')
x_train = dataset
x_train = x_train[:,1:x_train.shape[1]]

encoder = load_model("models/ae0.hdf5")
encoder_output = encoder.predict(x_train)

ALL_minus1 = []
TTUT = []
TTTI = []
TT = []
TI = []
ALL_not_minus1 = []

for i in range(label.shape[0]):
    if label[i] == 0:
        ALL_minus1.append(encoder_output[i, 0:2])
    if label[i] == 1:
        TTTI.append(encoder_output[i, 0:2])
    if label[i] == 2:
        TI.append(encoder_output[i, 0:2])
    if label[i] == 3:
        TTUT.append(encoder_output[i, 0:2])
    if label[i] == 4:
        TT.append(encoder_output[i, 0:2])
    if label[i] == 5:
        ALL_not_minus1.append(encoder_output[i, 0:2])


ALL_minus1 = np.array(ALL_minus1)
TTTI = np.array(TTTI)
TI = np.array(TI)
TTUT = np.array(TTUT)
TT = np.array(TT)
ALL_not_minus1 = np.array(ALL_not_minus1)
fig = plt.figure(figsize=(8, 16))
plt.subplot(2, 1, 1)
plt.scatter(encoder_output[:, 0], encoder_output[:, 1], c='#31A97D', s=5)
plt.title("Deep AutoEncoder"  , fontsize=14)
plt.xlabel('(a)',fontsize=14)
plt.tick_params(labelsize=14)

plt.subplot(2, 1, 2)
'''plt.scatter(TTTI[:, 0], TTTI[:, 1], c='#31A97D', s=3, label='TI=-1,TT=-1,UT≠-1')
plt.scatter(TI[:, 0], TI[:, 1], c='#3B4D86', s=3, label='TI=-1,TT≠-1,UT≠-1')
plt.scatter(TTUT[:, 0], TTUT[:, 1], c='#1200FF', s=3, label='TI≠-1,TT=-1,UT=-1')
plt.scatter(TT[:, 0], TT[:, 1], c='#F7E625', s=3, label='TI≠-1,TT=-1,UT≠-1')
plt.scatter(ALL_not_minus1[:, 0], ALL_not_minus1[:, 1], c='#006400', s=3, label='TI≠-1,TT≠-1,UT≠-1')
plt.scatter(ALL_minus1[:, 0], ALL_minus1[:, 1], c='#9F79EE', s=3, label='TI=-1,TT=-1,UT=-1')'''
db = DBSCAN(eps=0.9, min_samples=7).fit(encoder_output)
plt.scatter(encoder_output[:, 0], encoder_output[:, 1], c=db.labels_, s=5)
plt.xlabel('(b)',fontsize=14)
plt.tick_params(labelsize=14)
plt.title("DBSCAN_eps0.900_sam_7", fontsize=14)
plt.subplots_adjust(wspace=0.2, hspace=0.4)  #调整子图间距


plt.show()