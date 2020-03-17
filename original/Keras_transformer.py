from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf
from keras.optimizers import *
from keras.constraints import non_neg
import time

import combine
import five_show
import check
import LoadStruct
import ErrorShow
from pandas import DataFrame

try:
    from tqdm import tqdm
    # from dataloader import TokenList, pad_to_longest
# for transformer
except:
    pass


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


# It's safe to use a 1-d mask for self-attention
class ScaledDotProductAttention():
    def __init__(self, attn_dropout=0.1):
        self.dropout = Dropout(attn_dropout)

    def __call__(self, q, k, v, mask):  # mask_k or mask_qk
        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / temper)([q, k])  # shape=(batch, q, k)
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+9) * (1. - K.cast(x, 'float32')))(mask)
           # attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, dropout, mode=1):
        self.mode = mode
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head * d_k, use_bias=False)
            self.ks_layer = Dense(n_head * d_k, use_bias=False)
            self.vs_layer = Dense(n_head * d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention()
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, s[2] // n_head])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], s[2] // n_head])  # [n_head * batch_size, len_q, d_k]
                return x

            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
                return x

            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = [];
            attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head);
                attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        return outputs, attn


class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
        self.norm_layer = LayerNormalization()

    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.norm_layer(Add()([enc_input, output]))
        output = self.pos_ffn_layer(output)
        return output, slf_attn


class DecoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.enc_att_layer = MultiHeadAttention(n_head, d_model, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
        self.norm_layer1 = LayerNormalization()
        self.norm_layer2 = LayerNormalization()

    def __call__(self, dec_input, enc_output, self_mask=None, enc_mask=None, dec_last_state=None):
        if dec_last_state is None: dec_last_state = dec_input
        output, slf_attn = self.self_att_layer(dec_input, dec_last_state, dec_last_state, mask=self_mask)
        x = self.norm_layer1(Add()([dec_input, output]))
        output, enc_attn = self.enc_att_layer(x, enc_output, enc_output, mask=enc_mask)
        x = self.norm_layer2(Add()([x, output]))
        output = self.pos_ffn_layer(x)
        return output, slf_attn, enc_attn


def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb)
        for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


class SelfAttention():
    def __init__(self, d_model, d_inner_hid, n_head, layers=6, dropout=0.1):
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, dropout) for _ in range(layers)]

    def __call__(self, src_emb, src_seq, return_att=False, active_layers=999):
        if return_att: atts = []
        mask = Lambda(lambda x: K.cast(K.greater(x, 0), 'float32'))(src_seq)
        # mask = None
        x = src_emb
        for enc_layer in self.layers[:active_layers]:
            x, att = enc_layer(x, mask)
            if return_att: atts.append(att)
        x = Dense(1,  use_bias=False)(x)
        return (x, atts) if return_att else x


def GetPadMask(q, k):
    '''
    shape: [B, Q, K]
    '''
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2,1])
    return mask

def GetSubMask(s):
    '''
    shape: [B, Q, K], lower triangle because the i-th row should have i 1s.
    '''
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


class Decoder():
    def __init__(self, d_model, d_inner_hid, n_head, layers=6, dropout=0.1):
        self.layers = [DecoderLayer(d_model, d_inner_hid, n_head, dropout) for _ in range(layers)]

    def __call__(self, tgt_emb, tgt_seq, src_seq, enc_output, return_att=False, active_layers=999):
        x = tgt_emb
        '''self_pad_mask = Lambda(lambda x: GetPadMask(x, x))(tgt_seq)
        self_sub_mask = Lambda(GetSubMask)(tgt_seq)
        self_mask = Lambda(lambda x: K.minimum(x[0], x[1]))([self_pad_mask, self_sub_mask])
        enc_mask = Lambda(lambda x: GetPadMask(x[0], x[1]))([tgt_seq, src_seq])'''
        if return_att: self_atts, enc_atts = [], []
        for dec_layer in self.layers[:active_layers]:
            x, self_att, enc_att = dec_layer(x, enc_output, None, None)
            if return_att:
                self_atts.append(self_att)
                enc_atts.append(enc_att)
        return (x, self_atts, enc_atts) if return_att else x


class PosEncodingLayer:
    def __init__(self, max_len, d_emb):
        self.pos_emb_matrix = Embedding(max_len, d_emb, trainable=False, \
                                        weights=[GetPosEncodingMatrix(max_len, d_emb)])

    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def __call__(self, seq, pos_input=False):
        x = seq
        if not pos_input: x = Lambda(self.get_pos_seq)(x)
        return self.pos_emb_matrix(x)


def get_loss(y_true, y_pred):
    _loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), 1))
    return _loss


d_model = 6
d_inner_hid = 1
n_head = 4
layers = 6
dropout = 0.1
active_layers = 999
s2s_name = 'transformert.csv'
scores_name = 'tran_scoret.csv'
fig_name = 'transformert'


ld = LoadStruct.LoadData(0.9, 5, 0.01)
encoder_input = Input(shape=(5, 6), dtype='float32')
decoder_input = Input(shape=(5, 6), dtype='float32')

em_Dropout = Dropout(dropout)
pos_emb = PosEncodingLayer(5, 6)
add_layer = Lambda(lambda x: x[0] + x[1], output_shape=lambda x: x[0])

encoder = SelfAttention(d_model, d_inner_hid, n_head, layers, dropout)
enc_output = encoder(encoder_input, encoder_input, active_layers=active_layers)

decoder = Decoder(d_model, d_inner_hid, n_head, layers, dropout)
dec_output = decoder(encoder_input, decoder_input, decoder_input, enc_output, active_layers=active_layers)

target_layer = TimeDistributed(Dense(6, use_bias=False))
final_output = target_layer(dec_output)

loss = get_loss(decoder_input, final_output)
ppl = K.exp(loss)

pre_model = Model([encoder_input, decoder_input], final_output)
pre_model.add_loss([loss])
pre_model.compile(optimizer=Adam(0.001, 0.9, 0.98, epsilon=1e-9), loss=get_loss)
# pre_model.summary()
# for layers in pre_model:
#    print(layers.name)
# print(pre_model.get_config())
pre_model.metrics_names.append('ppl')
pre_model.metrics_tensors.append(ppl)
train_b = time.time()
pre_model.fit([ld.trainx, ld.trainx], ld.trainy,
              batch_size=512,
              epochs=50)
train_e = time.time()
train_t = train_e - train_b
print('train time = '+ str(train_t))

scores_model = Model(pre_model.get_layer('input_1').input, pre_model.get_layer('time_distributed_79').output)
out = pre_model.predict([ld.testx, ld.testx])
ErrorShow.draw(out.reshape(out.shape[0], 30), ld.testy.reshape(ld.testy.shape[0], 30), 'transformert')

x = np.concatenate((ld.trainx, ld.testx), axis=0)
pre_out = pre_model.predict([x, x])
s1b = time.time()
scores = scores_model.predict(x)
s1e = time.time()
print('score time = %f' %(s1e-s1b))

s2b = time.time()
scores2 = scores_model.predict(pre_out)
s2e = time.time()
print('score time = %f' %(s2e-s2b))


save = DataFrame(scores.reshape([scores.shape[0], scores.shape[1]*scores.shape[2]]))
save.to_csv(s2s_name, header=True, index=False)
save2 = DataFrame(scores2.reshape([scores2.shape[0], scores2.shape[1]*scores2.shape[2]]))
save2.to_csv('pre_'+s2s_name, header=True, index=False)

combine.combine(s2s_name, 'pre_'+s2s_name, scores_name)
five_show.draw(scores_name, fig_name)
for name in ['s2s', 'pres2s']:
    print(name+'checking...')
    check.check(name, scores_name)#
