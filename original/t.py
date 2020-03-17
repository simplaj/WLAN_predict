import tensorflow as tf

with tf.Session() as sess:
   x = [[[1, 2, 3],
         [4, 5, 6]],
        [[7, 8, 9],
         [10, 11, 12]],
        [[13, 14, 15],
         [16, 17, 18]]]
   y = tf.transpose(x, [1, 0, 2])
   z = y.eval()
print(z[-1])
