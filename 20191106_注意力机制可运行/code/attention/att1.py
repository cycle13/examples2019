# -*- coding: utf-8 -*-
"""
mnist attention
"""
from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam

os.environ["CUDA_VISIBLE_DEVICES"] = "1" #"1,0"

TIME_STEPS = 28
INPUT_DIM = 28
lstm_units = 64
 
# data pre-processing
(X_train, y_train), (X_test, y_test) = mnist.load_data('mnist.npz')
X_train = X_train.reshape(-1, 28, 28) / 255.
X_test = X_test.reshape(-1, 28, 28) / 255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
 
# first way attention
#https://blog.csdn.net/uhauha2929/article/details/80733255
#相似度函数采用的是一层全连接层。全连接层的输出经过softmax激活函数计算权重。
#他对隐层向量的每一维在每个时间步上进行了softmax操作，这里函数的返回值是三维的，也就是说这里只是乘上了权重，但并没有求和。
#注意：这里直接对经过线性变换后的所有向量进行了softmax，并没有乘上context vector后再做，好像在分类例子里影响不大。
def attention_3d_block(inputs):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1),name='att_input')(inputs)
    a = Dense(TIME_STEPS, activation='softmax',name='att_dense')(a)
    a_probs = Permute((2, 1), name='att_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='att_mul')
    return output_attention_mul
 
# build RNN model with attention
inputs = Input(shape=(TIME_STEPS, INPUT_DIM),name='input_input')
drop1 = Dropout(0.3,name='input_drop')(inputs)
lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm')(drop1)
attention_mul = attention_3d_block(lstm_out)
attention_flatten = Flatten(name='att_flat')(attention_mul)
drop2 = Dropout(0.3,name='out_drop')(attention_flatten)
output = Dense(10, activation='sigmoid',name='out_dense')(drop2)
model = Model(inputs=inputs, outputs=output)
 
# second way attention
# inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
# units = 32
# activations = LSTM(units, return_sequences=True, name='lstm_layer')(inputs)
#
# attention = Dense(1, activation='tanh')(activations)
# attention = Flatten()(attention)
# attention = Activation('softmax')(attention)
# attention = RepeatVector(units)(attention)
# attention = Permute([2, 1], name='attention_vec')(attention)
# attention_mul = merge([activations, attention], mode='mul', name='attention_mul')
# out_attention_mul = Flatten()(attention_mul)
# output = Dense(10, activation='sigmoid')(out_attention_mul)
# model = Model(inputs=inputs, outputs=output)
 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
 
print('Training------------')
model.fit(X_train, y_train, epochs=100, batch_size=16,verbose=2)
 
print('Testing--------------')
loss, accuracy = model.evaluate(X_test, y_test)
 
print('test loss:', loss)
print('test accuracy:', accuracy)