# -*- coding: utf-8 -*-
from __future__ import print_function
"""
haikou layernorm
fun:
	save csv
env:
	Win7 64bit;anaconda 1.7.2;python 3.5;tensorflow1.10.1;Keras2.2.4
	pip3,matplotlib2.2.3,seaborn0.9.0
"""
__author__ = 'elesun'
import keras
from keras_layer_normalization import LayerNormalization

print("test layer_norm")
input_layer = keras.layers.Input(shape=(2, 3),name="in")
norm_layer = LayerNormalization(name="norm")(input_layer)
model = keras.models.Model(inputs=input_layer, outputs=norm_layer)
model.compile(optimizer='adam', loss='mse', metrics={})
model.summary()