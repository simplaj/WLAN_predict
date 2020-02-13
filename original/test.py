import tensorflow as tf
from keras.layers import Dot, Activation, Concatenate
from keras import backend as K
import numpy as np
a = np.arange(19071*5*6).reshape(19071, 5, 6) # a和b的维度有些讲究，具体查看Dot类的build方法
b = np.arange(19071*5*6).reshape(19071, 5, 6)
output = K.batch_dot(K.constant(a), K.constant(b),  axes=[2, 2])
_output = Activation('softmax')(output)
__output = Dot(axes=[2, 1])([_output, K.constant(b)])
___output = Concatenate(axis=-1)([__output, K.constant(b)])
with tf.Session() as sess:
    output_array = sess.run(__output)
    print( output_array)
    print( output_array.shape)
