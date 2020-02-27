from __future__ import print_function
import tensorflow as tf
import keras
import keras as K
from keras.constraints import non_neg
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dot, Activation, Concatenate
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import LoadStruct
import ErrorShow
from sklearn.metrics import mean_squared_error, mean_absolute_error

batch_size = 1024  # Batch size for training.
epochs = 5000  # Number of epochs to train for.
latent_dim = 1 # Latent dimensionality of the encoding space.
# num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
e_lens = 6
d_lens = 6
steps = 5
ld = LoadStruct.LoadData(0.8, steps, 0.01)
len_data = len(ld.trainx)
encoder_input_data = np.zeros(
    (len_data, steps, e_lens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len_data, steps, d_lens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len_data, d_lens),
    dtype='float32')

encoder_input_data = ld.trainx
decoder_target_data = ld.trainy

encoder_inputs = Input(shape=(None, e_lens))
encoder1 = LSTM(latent_dim, return_sequences=True)
encoder_outputs = encoder1(encoder_inputs)
encoder2 = LSTM(latent_dim, return_state=True,)
en_outputs, en_h, en_c = encoder2(encoder_outputs)

decoder_dense = Dense(d_lens, activation='softmax')
en_outputs = decoder_dense(en_outputs)
model = Model(encoder_inputs, en_outputs)
Ada = K.optimizers.Adadelta(lr=0.08)


def mse2(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), 1))


model.compile(optimizer=Ada, loss=mse2)

history = model.fit(encoder_input_data, decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
outputs = model.predict(ld.testx)
len_p = len(ld.testx)
ErrorShow.draw(outputs, ld.testy, 1, 1, 1)
'''_save = outputs.reshape((outputs.shape[0], outputs.shape[1]*outputs.shape[2]))
_y = decoder_target_data.reshape((decoder_target_data.shape[0], decoder_target_data.shape[1]*decoder_target_data.shape[2]))
_mse = mean_squared_error(_y, _save)
_mae = mean_absolute_error(_y, _save)
print("mse: %f" % _mse)
print("mae: %f" % _mae)
__save = DataFrame(_save)
__save.columns = ['0', '0', '0', '0', '0', '0', '1', '1', '1', '1', '1',
                  '1', '2', '2', '2', '2', '2', '2', '3',
                  '3', '3', '3', '3', '3', '4', '4', '4', '4', '4', '4']
__save.to_csv('pre/Dot.csv', index=False, header=True)'''

