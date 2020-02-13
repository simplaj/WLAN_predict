from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
import pandas as pd
from pandas import DataFrame

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
        source_seq = [docs_source[p, 0]] + [docs_source[p, 1]]+[docs_source[p, 2]] \
                     + [docs_source[p, 3]]+[docs_source[p, 4]] + [docs_source[p, 5]]
        # if i > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
           # decoder_target_data[p, i - 1, :] = source_seq
        encoder_input_data[p, i, :] = source_seq
        decoder_input_data[p, i, :] = source_seq
        decoder_target_data[p, i, :] = source_seq
encoder_model = load_model('models/s2s_n_512.h5')
model = load_model('models/s2s_Dot_att_pre.h5')

scores_outputs = encoder_model.predict(encoder_input_data)
output = scores_outputs.reshape((scores_outputs.shape[0], scores_outputs.shape[1]))
save = DataFrame(output)
save.to_csv('scores/s2s_Dot_att.csv', index=False, header=True)

outputs = model.predict([encoder_input_data, decoder_input_data])
# _outputs = outputs.reshape((outputs.shape[0], outputs.shape[1], outputs.shape[2]))
# _save = _outputs[ :, 0, :]
_save = outputs.reshape((outputs.shape[0], outputs.shape[1]*outputs.shape[2]))
print(_save.shape)
__save = DataFrame(_save)
__save.to_csv('pre/Dot_att.csv', index=False, header=True)
# np.savetxt('pre/Dot_att.csv', _outputs, delimiter=',')
