from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.utils import to_categorical
from pandas import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import *


def get_random_block_from_data(data,batch_size):
    start_index=np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

batch_size = 2560
latent_dim = 2
intermediate_dim = 256
epochs = 200
epsilon_std = 1.0
neg = 0.3

dataframe = read_csv('data/addDim51.csv')
dataset = dataframe.values
dataset = dataset.astype('float32')
orignal_data = dataset
from keras.constraints import non_neg

dataframe = read_csv('data/normalization.csv')
dataset = dataframe.values
dataset = dataset.astype('float32')


label = []
'''
dataset[i,2] = TI
dataset[i,3] = TT
dataset[i,4] = UT
'''


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
dataset = dataset[:, 2:dataset.shape[1]]
x_train = dataset
original_dim = x_train.shape[1]

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim,activation='tanh',kernel_constraint= non_neg())(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
decoder_h = Dense(intermediate_dim,activation='sigmoid',kernel_constraint= non_neg())
decoder_out = Dense(original_dim,kernel_constraint= non_neg())
h_decoded = decoder_h(z)
x_decoded_mean = decoder_out(h_decoded)
vae = Model(x, x_decoded_mean)

mse_loss = K.sum(K.square(x-x_decoded_mean),axis= -1)
binary_crossentropy_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
xent_loss = mse_loss
vae_loss = K.mean(4*xent_loss+ kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
vae.fit(x_train,shuffle=True,epochs=epochs,batch_size=batch_size,validation_data=(x_train, None),verbose= 2)
encoder = Model(x, z_mean)
x_test_encoded = encoder.predict(x_train)
encoder_output = vae.predict(x_train)
encoder_mse = mean_squared_error(x_train ,encoder_output)
encoder_mae = mean_absolute_error(x_train,encoder_output)
print("MSE = %.4f , MAE = %.4f"%(encoder_mse,encoder_mae))
db = DBSCAN(eps=0.007, min_samples=5).fit(x_test_encoded)
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=db.labels_,s = 5)
plt.title("R, neg = %.2f , dim = %d , epchos = %d , mse = %.4f , mae = %.4f" % (neg,original_dim,epochs, encoder_mse,encoder_mae))
plt.show()
vae.save_weights('test.hdf5')
output = np.append(x_test_encoded,orignal_data,axis=-1)
output = np.append(output,label.reshape(-1,1),axis = -1)
save = DataFrame(output)
save.columns = ['x','y','SP','RP','TI','TT','UT','LMS','CML','MLR','Label']
save.to_csv('scores/outputs_new.csv',index=False,header=True)
encoder_result = x_test_encoded
RP0 = []
ALL_minus1 = []
TTUT = []
TTTI = []
TT = []
UTTI = []
UT = []
TI = []
ALL_not_minus1 = []
for i in range(label.shape[0]):
    if label[i] == 0:
        ALL_minus1.append(x_test_encoded[i,0:2])
    if label[i] == 1:
        TTTI.append(x_test_encoded[i,0:2])
    if label[i] == 2:
        TI.append(x_test_encoded[i,0:2])
    if label[i] == 3:
        TTUT.append(x_test_encoded[i, 0:2])
    if label[i] == 4:
        TT.append(x_test_encoded[i, 0:2])
    if label[i] == 5:
        ALL_not_minus1.append(x_test_encoded[i, 0:2])
ALL_minus1 = np.array(ALL_minus1)
TTTI = np.array(TTTI)
TI = np.array(TI)
TTUT = np.array(TTUT)
TT = np.array(TT)
ALL_not_minus1 = np.array(ALL_not_minus1)
#plt.scatter(RP0[0][0], RP0[0][1], c='#306B8B', s=3, label='RP=0')
plt.scatter(ALL_minus1[:,0], ALL_minus1[:,1], c='#9F79EE', s=3, label='TT=-1,UT=-1,TI=-1')
plt.scatter(TTUT[:,0], TTUT[:,1], c='#1200FF', s=3, label='TT=-1,UT=-1,TI≠-1')
plt.scatter(TTTI[:,0], TTTI[:,1], c='#31A97D', s=3, label='TT=-1,UT≠-1,TI=-1')
plt.scatter(TT[:,0], TT[:,1], c='#F7E625', s=3, label='TT=-1,UT≠-1,TI≠-1')
#plt.scatter(UTTI[0][0], UTTI[0][1], c='#306A8B', s=3, label='TT≠-1,UT=-1,TI=-1')
#plt.scatter(UT[0][0], UT[0][1], c='#430D5E', s=3, label='TT≠-1,UT=-1,TI≠-1')
plt.scatter(TI[:,0], TI[:,1], c='#3B4D86', s=3, label='TT≠-1,UT≠-1,TI=-1')
plt.scatter(ALL_not_minus1[:,0], ALL_not_minus1[:,1], c='#006400', s=3, label='TT≠-1,UT≠-1,TI≠-1')
plt.title("R , neg = %.2f , dim = %d , epchos = %d , mse = %.4f , mae = %.4f" % (neg,original_dim, epochs, encoder_mse, encoder_mae))
plt.legend(bbox_to_anchor=(-0.02, -0.12), loc='center left', ncol=3, borderaxespad=0, markerscale=4, fontsize=8)
plt.show()
