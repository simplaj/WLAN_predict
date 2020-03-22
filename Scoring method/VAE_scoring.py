from __future__ import print_function
import numpy as np
from keras.constraints import non_neg
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
import time


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]


batch_size = 2560
latent_dim = 1
intermediate_dim = 256
epochs = 200
epsilon_std = 1.0
neg = 0.3

dataframe = read_csv('../data/addDim51.csv')
dataset = dataframe.values
dataset = dataset.astype('float32')
orignal_data = dataset

dataframe = read_csv('../data/normalization.csv')
dataset = dataframe.values
dataset = dataset.astype('float32')


label = []
'''
dataset[i,2] = TI
dataset[i,3] = TT
dataset[i,4] = UT
'''


for i in range(orignal_data.shape[0]):
    if orignal_data[i, 2] == -1 and orignal_data[i, 3] == -1 and orignal_data[i, 4] == -1:
        label.append(0)
    if orignal_data[i, 2] == -1 and orignal_data[i, 3] == -1 and orignal_data[i, 4] != -1:
        label.append(1)
    if orignal_data[i, 2] == -1 and orignal_data[i, 3] != -1 and orignal_data[i, 4] != -1:
        label.append(2)
    if orignal_data[i, 2] != -1 and orignal_data[i, 3] == -1 and orignal_data[i, 4] == -1:
        label.append(3)
    if orignal_data[i, 2] != -1 and orignal_data[i, 3] == -1 and orignal_data[i, 4] != -1:
        label.append(4)
    if orignal_data[i, 2] != -1 and orignal_data[i, 3] != -1 and orignal_data[i, 4] != -1:
        label.append(5)
label = np.array(label)
dataset = dataset[:, 2:dataset.shape[1]]
x_train = dataset
original_dim = x_train.shape[1]
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='tanh', kernel_constraint=non_neg())(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
decoder_h = Dense(intermediate_dim, activation='sigmoid', kernel_constraint=non_neg())
decoder_out = Dense(original_dim, kernel_constraint=non_neg())
h_decoded = decoder_h(z)
x_decoded_mean = decoder_out(h_decoded)
vae = Model(x, x_decoded_mean)

mse_loss = K.sum(K.square(x-x_decoded_mean), axis=-1)
binary_crossentropy_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
xent_loss = mse_loss
vae_loss = K.mean(4*xent_loss+ kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
vaeb = time.time()
vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_train, None),
        verbose=2)
vaee = time.time()
print('train time = %f' %(vaee - vaeb))
encoder = Model(vae.get_layer('input_1').input, vae.get_layer('dense_2').output)
vaeb2 = time.time()
x_test_encoded = encoder.predict(x_train)
vaee2 = time.time()
print('score time = %f'%(vaee2 - vaeb2))
encoder_output = vae.predict(x_train)
encoder_mse = mean_squared_error(x_train, encoder_output)
encoder_mae = mean_absolute_error(x_train, encoder_output)
print("MSE = %.4f , MAE = %.4f" % (encoder_mse, encoder_mae))
db = DBSCAN(eps=0.007, min_samples=5).fit(x_test_encoded)
# vae.save_weights('test21.hdf5')
save = DataFrame(x_test_encoded)
save.columns = ['score']
save.to_csv('../scores/score_vae.csv', index=False, header=True)
