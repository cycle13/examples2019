# -*- coding: utf-8 -*-
"""
mnist attention
"""
from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.layers import merge

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
 
# second way attention
inputs = Input(shape=(TIME_STEPS, INPUT_DIM),name='input_input')
units = 32
activations = LSTM(units, return_sequences=True, name='lstm_layer')(inputs)
attention = Dense(1, activation='tanh',name='att_dense')(activations)
attention = Flatten(name='att_flat')(attention)
attention = Activation('softmax',name='att_act')(attention)
attention = RepeatVector(units,name='att_repeat')(attention)
attention = Permute([2, 1], name='att_vec')(attention)
#attention_mul = merge([activations, attention], mode='mul', name='att_mul')
attention_mul = multiply([activations, attention], name='att_mul')
out_attention_mul = Flatten(name='out_flat')(attention_mul)
output = Dense(10, activation='sigmoid',name='out_dense')(out_attention_mul)
model = Model(inputs=inputs, outputs=output)
 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
 
print('Training------------')
model.fit(X_train, y_train, epochs=100, batch_size=16,verbose=2)
 
print('Testing--------------')
loss, accuracy = model.evaluate(X_test, y_test)
 
print('test loss:', loss)
print('test accuracy:', accuracy)