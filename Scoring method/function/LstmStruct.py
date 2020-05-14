# -*- coding=utf-8 -*-

import numpy as np
import tensorflow as tf
from keras.layers import Dot, Activation, Concatenate, Lambda, Dropout
from keras import backend as K
import time


class LstmData():
    tf.reset_default_graph()

    def __init__(self, ld, lstm_layers, lstm_hidden_size, timeSteps, epochs, lr):
        self.timeSteps = timeSteps  # LSTM输入序列数据长度
        self.lstm_layers = lstm_layers  # LSTM隐藏层层数
        self.epochs = epochs  # 训练循环周期
        self.ld = ld
        self.lstm_input_size = 14  # LSTM输入数据的维度
        self.lstm_hidden_size = lstm_hidden_size  # LSTM隐藏层节点数
        self.final_out_size = 14  # 预测数据的维度
        self.lr = lr
        # self.states = states
        '''
        self.ld.trainx = tf.convert_to_tensor(self.ld.trainx)
        attention = Dot(axes=[2, 2])([self.ld.trainx, self.ld.trainx])
        attention = Activation('softmax')(attention)
        context = Dot(axes=[2, 1])([attention, self.ld.trainx])
        self.ld.trainx = Concatenate(axis=-1)([context, self.ld.trainx])



        self.ld.testx = tf.convert_to_tensor(self.ld.testx)
        attention = Dot(axes=[2, 2])([self.ld.testx, self.ld.testx])
        attention = Activation('softmax')(attention)
        context = Dot(axes=[2, 1])([attention, self.ld.testx])
        self.ld.testx = Concatenate(axis=-1)([context, self.ld.testx])

        with tf.Session() as sess:
            self.ld.trainx = self.ld.trainx.eval()
            self.ld.testx = self.ld.testx.eval()
        '''

    def attn(self, q, k, v):
        self.dropout = Dropout(0.1)
        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
        attn = K.batch_dot(q, k, axes=[2, 2]) / temper
        attn = tf.nn.softmax(attn, axis=1)
        # attn = self.dropout(attn)

        output = K.batch_dot(attn, v)
        # hidden: B * D
        # encoder_outputs: B * S * D
        '''attn_weights = tf.matmul(encoder_outputs, tf.expand_dims(hidden, 2))

        # attn_weights: B * S * 1
        attn_weights = tf.nn.softmax(attn_weights, axis=1)
        context = tf.squeeze(tf.matmul(tf.transpose(encoder_outputs, [0, 2, 1]), attn_weights))
        # context: B * D'''
        return output

    # 搭建LSTM结构
    def multi_layer_dynamic_lstm(self, input_x):
        encoder_cell = tf.contrib.rnn.LSTMCell(self.lstm_hidden_size)
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
            encoder_cell, input_x,
            dtype=tf.float32,
        )

        '''hidden = tf.transpose(encoder_final_state, [1, 0, 2])
        hidden = hidden[0]
        '''

        # input = self.attn(input_x, input_x, input_x)

        decoder_cell = tf.contrib.rnn.LSTMCell(self.lstm_hidden_size)
        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
            decoder_cell, input_x,
            initial_state=encoder_final_state,
            dtype=tf.float32, scope="plain_decoder",
        )
        hiddens = tf.transpose(decoder_outputs, [1, 0, 2])

        return decoder_outputs, encoder_outputs
        # 返回训练数据的预测结果，以及测试数据的结果

    def lstmWork(self):
        # ——————————————————定义神经网络变量——————————————————
        X = tf.placeholder(tf.float32, [None, self.timeSteps, self.lstm_input_size])
        y = tf.placeholder(tf.float32, [None, self.timeSteps, self.final_out_size])
        #       batch_size = []
        #       initial_state = tf.placeholder(tf.float32, [batch_size, None])
        with tf.variable_scope("rcnn", reuse=None):
            train_lstm_out, scores = self.multi_layer_dynamic_lstm(X)
            # 将LSTM最后一层最后一个时刻的隐藏层向量，通过全连接层输出8维向量，作为下一个时刻的预测值
            trainPred = tf.contrib.layers.fully_connected(inputs=train_lstm_out, num_outputs=self.final_out_size,
                                                          activation_fn=tf.nn.sigmoid)
        with tf.variable_scope("rcnn", reuse=True):
            test_lstm_out, t_scores = self.multi_layer_dynamic_lstm(X)
            # 将LSTM最后一层最后一个时刻的隐藏层向量，通过全连接层输出8维向量，作为下一个时刻的预测值
            testPred = tf.contrib.layers.fully_connected(inputs=test_lstm_out, num_outputs=self.final_out_size,
                                                         activation_fn=tf.nn.sigmoid)
        # 损失函数以及优化器设置
        x = tf.Variable(np.random.random((5, 10)))
        mse = tf.reduce_mean(tf.reduce_sum(tf.square(y - trainPred), 1))  # 每个样本欧式距离计算预测向量和实际向量的距离，然后求均值
        train_op = tf.train.AdadeltaOptimizer(self.lr).minimize(mse)
        # w_noneg_op = tf.assign(x, tf.clip_by_value(x, 0, np.infty))
        # 正式过程
        with tf.Session() as sess:
            loss = []
            sess.run(tf.global_variables_initializer())
            stt = time.time()
            for i in range(self.epochs):
                _, mse_ = sess.run([train_op, mse], feed_dict={X: self.ld.trainx, y: self.ld.trainy})
                # _, msee_ = sess.run([w_noneg_op, mse], feed_dict={X: self.ld.trainx, y: self.ld.trainy})
                loss.append(mse_)
                print(i, mse_)
            ent = time.time()
            print('train time = %f' % (ent - stt))
            pred2 = sess.run(testPred, feed_dict={X: np.concatenate((self.ld.trainx, self.ld.testx), axis=0),
                                                  y: np.concatenate((self.ld.trainy, self.ld.testy))})
            ht = time.time()
            hid = sess.run(t_scores, feed_dict={X: np.concatenate((self.ld.trainx, self.ld.testx), axis=0),
                                                y: np.concatenate((self.ld.trainy, self.ld.testy))})
            he = time.time()
            print('score time = %f' % (he - ht))
            hid2 = sess.run(t_scores, feed_dict={X: pred2, y: pred2})
        return pred2, np.concatenate((self.ld.trainy, self.ld.testy)), hid, loss, hid2
