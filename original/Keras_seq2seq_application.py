from __future__ import print_function

from keras.constraints import non_neg
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import pandas as pd
from pandas import DataFrame

batch_size = 512  # Batch size for training.
epochs = 200  # Number of epochs to train for.
latent_dim = 1  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
dataframe = pd.read_csv("data/normalization.csv")
len_data = DataFrame(dataframe).shape[0]
data = dataframe.values
docs_source = data[:, 2:]
docs_target = data[:, 2:]
e_lens = 6
d_lens = 6
steps = 5
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
        source_seq = [docs_source[p, 0]] + [docs_source[p, 1]]+[docs_source[p, 2]]\
                     + [docs_source[p, 3]]+[docs_source[p, 4]] + [docs_source[p, 5]]
        encoder_input_data[p, i, :] = source_seq
        decoder_input_data[p, i, :] = source_seq
        decoder_target_data[p, i, :] = source_seq


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, e_lens))
encoder = LSTM(latent_dim, return_sequences=True, return_state=True, kernel_constraint=non_neg())
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, d_lens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, kernel_constraint=non_neg())
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(d_lens, activation='softmax', kernel_constraint=non_neg())
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

# Save model
model.save('models/s2s_n_512.h5')
