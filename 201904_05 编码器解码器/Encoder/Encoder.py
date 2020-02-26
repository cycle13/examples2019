# -*- coding: utf-8 -*-
""" 
encoder_mnist
fun:
	在MNIST数据集上，实现特征压缩和特征解压
	将编码得到的低维“特征值”在低维空间中可视化出来，直观显示数据的聚类效果。
env:
	supermapBJAIGPU Ubuntu 16.04.5 LTS,Linux version 4.15.0-43-generic;
	python 2.7;pip2 list;numpy1.16.1;pandas0.24.1;matplotlib2.2.3;
	tensorflow-gpu1.12.0;Keras2.2.4

"""
from __future__ import print_function
import tensorflow as tf  
import matplotlib.pyplot as plt  
  
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
import input_data
mnist = input_data.read_data_sets("data/", one_hot=False)

learning_rate = 0.01  
training_epochs = 10000
batch_size = 256  
display_step = 1  
n_input = 784  
X = tf.placeholder("float", [None, n_input])  
  
n_hidden_1 = 128  
n_hidden_2 = 64  
n_hidden_3 = 10  
n_hidden_4 = 2  
weights = {  
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],)),  
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)),  
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)),  
    'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],)),  
    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)),  
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)),  
    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)),  
    'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input],)),  
}  
biases = {  
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),  
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),  
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),  
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),  
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),  
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),  
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),  
    'decoder_b4': tf.Variable(tf.random_normal([n_input])),  
}  
def encoder(x):  
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),  
                                   biases['encoder_b1']))  
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),  
                                   biases['encoder_b2']))  
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),  
                                   biases['encoder_b3']))  
    # 为了便于编码层的输出，编码层随后一层不使用激活函数  
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),  
                                    biases['encoder_b4'])  
    return layer_4  
  
def decoder(x):  
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),  
                                   biases['decoder_b1']))  
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),  
                                   biases['decoder_b2']))  
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),  
                                biases['decoder_b3']))  
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),  
                                biases['decoder_b4']))  
    return layer_4  
  
encoder_op = encoder(X)  
decoder_op = decoder(encoder_op)  
  
y_pred = decoder_op  
y_true = X  
  
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))  
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)  
  
with tf.Session() as sess:  
    # tf.initialize_all_variables() no long valid from  
    # 2017-03-02 if using tensorflow >= 0.12  
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  
        init = tf.initialize_all_variables()  
    else:  
        init = tf.global_variables_initializer()  
    sess.run(init)  
    total_batch = int(mnist.train.num_examples/batch_size)  
    for epoch in range(training_epochs):  
        for i in range(total_batch):  
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0  
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})  
        if epoch % display_step == 0:  
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))  
    print("Optimization Finished!")  
  
    encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})  
    plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)  
    plt.colorbar()  
    plt.show()  
