from keras.models import Model, load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.layers import Input
import numpy as np
import pandas as pd
from pandas import DataFrame

dataframe = pd.read_csv("data/normalization.csv")
len_ = DataFrame(dataframe).shape[0]
len_data = 6000
data = dataframe.values
docs_source = data[13000:, 2:]
docs_target = data[13000:, 2:]
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
            source_seq = [docs_source[p + i, 0]] + [docs_source[p + i, 1]] + [docs_source[p + i, 2]] \
                         + [docs_source[p + i, 3]] + [docs_source[p + i, 4]] + [docs_source[p + i, 5]]
            # target_seq = [docs_source[p + i+1, 0]] + [docs_source[p + i+1, 1]] + [docs_source[p + i+1, 2]] \
            #            + [docs_source[p + i+1, 3]] + [docs_source[p + i+1, 4]] + [docs_source[p + i+1, 5]]
            encoder_input_data[p, i, :] = source_seq
            decoder_input_data[p, i, :] = source_seq
            decoder_target_data[p, i, :] = source_seq

encoder_model = load_model('models/Dot_att_0.h5')
model = load_model('models/Dot_att_pre_0.h5')

scores_outputs = encoder_model.predict(encoder_input_data)
output = scores_outputs.reshape((scores_outputs.shape[0], scores_outputs.shape[1]))
save = DataFrame(output)
save.to_csv('scores/Dot_att_0.csv', index=False, header=True)



outputs = model.predict([encoder_input_data, decoder_input_data])
# _outputs = outputs.reshape((outputs.shape[0], outputs.shape[1], outputs.shape[2]))
# _save = _outputs[ :, 0, :]
_save = outputs.reshape((outputs.shape[0], outputs.shape[1]*outputs.shape[2]))
_y = decoder_target_data.reshape((decoder_target_data.shape[0], decoder_target_data.shape[1]*decoder_target_data.shape[2]))
_mse = mean_squared_error(_y, _save)
_mae = mean_absolute_error(_y, _save)
print("mse: %f" % _mse)
print("mae: %f" % _mae)
__save = DataFrame(_save)
__save.columns = ['0', '0', '0', '0', '0', '0', '1', '1', '1', '1', '1',
                '1', '2', '2', '2', '2', '2', '2', '3',
                '3', '3', '3', '3', '3', '4', '4', '4', '4', '4', '4']
__save.to_csv('pre/Dot_att_0.csv', index=False, header=True)
# np.savetxt('pre/Dot_att.csv', _outputs, delimiter=',')
