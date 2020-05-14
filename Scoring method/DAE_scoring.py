from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import preprocessing
from keras.layers import Activation, Dense, Input
from keras.layers import Conv1D, Flatten
from keras.layers import Reshape
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd
from keras.constraints import non_neg
import time

# read dataset
from pandas import DataFrame

dataFrame = pd.read_csv('../data/simdata.csv')
dataset = dataFrame.values
dataset = dataset.astype('float32')
original_data = dataset
dataset = dataset[:, 1:]
dataset = np.expand_dims(dataset, axis=2)
x_train = dataset
x_test = x_train

# Generate corrupted MNIST images by adding noise with normal dist
# centered at 0.5 and std=0.5
noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
x_test_noisy = x_test + noise

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


original_dim = x_train.shape[1]
# Network parameters
input_shape = (original_dim, 1)
batch_size = 5120
kernel_size = 6
latent_dim = 1
# Encoder/Decoder number of CNN layers and filters per layer
layer_filters = [original_dim, original_dim]

# Build the Autoencoder Model
# First build the Encoder Model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# Stack of Conv2D blocks
# Notes:
# 1) Use Batch Normalization before ReLU on deep networks
# 2) Use MaxPooling2D as alternative to strides>1
# - faster but not as good as strides>1
for filters in layer_filters:
    x = Conv1D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)

# Shape info needed to build Decoder Model
shape = K.int_shape(x)

# Generate the latent vector
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector', kernel_constraint=non_neg())(x)

# Instantiate Encoder Model
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

latent_inputs = Input(shape=(latent_dim, 1), name='decoder_input')
x = Dense(original_dim*1, kernel_constraint=non_neg())(latent_inputs)
x = Reshape((original_dim, 1))(x)

outputs = Activation('sigmoid', name='decoder_output')(x)

# Instantiate Decoder Model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# Autoencoder = Encoder + Decoder
# Instantiate Autoencoder Model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

autoencoder.compile(loss='mse', optimizer='adam')
daeb = time.time()
# Train the autoencoder
autoencoder.fit(x_train_noisy,
                x_train,
                validation_data=(x_test_noisy, x_test),
                epochs=200,
                batch_size=batch_size)
daee = time.time()
print('train time = %f' %(daee-daeb))
# Predict the Autoencoder output from corrupted test images
daeb1 = time.time()
x_score = encoder.predict(x_train)
daee1 = time.time()
print('score time = %f'%(daee1 - daeb1))
# min_max_scaler = preprocessing.MinMaxScaler()
# x_minmax = min_max_scaler.fit_transform(x_score)
# output_dae = np.append(x_score, original_data, axis=-1)
save = DataFrame(x_score)
save.columns = ['score_dae']
save.to_csv('../scores/score_dae_s2.csv', index=False, header=True)
