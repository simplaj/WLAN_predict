from __future__ import print_function

import keras
import keras as K
from keras.constraints import non_neg
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dot, Activation, Concatenate
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

batch_size = 4  # Batch size for training.
epochs = 200  # Number of epochs to train for.
latent_dim = 1  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
dataframe = pd.read_csv("data/normalization.csv")
len_data = 13000
data = dataframe.values
docs_source = data[:13500, 2:]
docs_target = data[:13500, 2:]
e_lens = 6
d_lens = 6
steps = 1
encoder_input_data = np.zeros(
    (len_data, steps, e_lens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len_data, steps, d_lens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len_data, steps, d_lens),
    dtype='float32')

for p in range(len_data):
    for i in range(steps):  # 输入五行数据
            source_seq = [docs_source[p + i, 0]] + [docs_source[p + i, 1]] + [docs_source[p + i, 2]] \
                         + [docs_source[p + i, 3]] + [docs_source[p + i, 4]] + [docs_source[p + i, 5]]
            # target_seq = [docs_source[p + i+1, 0]] + [docs_source[p + i+1, 1]] + [docs_source[p + i+1, 2]] \
              #         + [docs_source[p + i+1, 3]] + [docs_source[p + i+1, 4]] + [docs_source[p + i+1, 5]]
            encoder_input_data[p, i, :] = source_seq
            decoder_input_data[p, i, :] = source_seq
            decoder_target_data[p, i, :] = source_seq

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, e_lens))
encoder = LSTM(latent_dim,  return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, d_lens))

'''

# Attention
attention = Dot(axes=[2, 2])([decoder_inputs, encoder_inputs])
attention = Activation('softmax')(attention)
context = Dot(axes=[2, 1])([attention, encoder_inputs])
decoder_combined_context = Concatenate(axis=-1)([context, decoder_inputs])
'''
'''
# Attention
_Q = tf.convert_to_tensor(decoder_target_data)
_K = tf.convert_to_tensor(decoder_input_data)
_V = tf.convert_to_tensor(encoder_input_data)
decoder_combined_context = Attention(8, [5, 6], mask_right=False)([_Q, _K, _V])
'''

# Decoder
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(d_lens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='adamax', loss='mean_absolute_error',
              metrics=['mse', 'acc'])
history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

model.save('models/Dot_att_pre_0.h5')
encoder_model = Model(encoder_inputs, encoder_outputs)
# Save model
encoder_model.save('models/Dot_att_0.h5')
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()