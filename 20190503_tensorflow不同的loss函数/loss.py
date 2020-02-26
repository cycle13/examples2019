from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

a = tf.constant([[4.0, 4.0, 4.0], [3.0, 3.0, 3.0], [1.0, 1.0, 1.0]])
b = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
print(a)
print(b)

loss1 = tf.reduce_mean(tf.square(a - b))
loss2 = tf.losses.mean_squared_error(a, b)
loss3 = tf.reduce_mean(tf.squared_difference(a, b))
loss4 = tf.sqrt(tf.reduce_mean(tf.square(a - b)))
loss5 = tf.reduce_mean(tf.abs(a - b))
loss6 = tf.nn.l2_loss(a - b)#  sum(t ** 2) / 2
loss7 = tf.reduce_sum(tf.square(a - b))/2#  sum((a-b) ** 2) / 2

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(loss1))
    print(sess.run(loss2))
    print(sess.run(loss3))
    print(sess.run(loss4))
    print(sess.run(loss5))
    print(sess.run(loss6))
    print(sess.run(loss7))