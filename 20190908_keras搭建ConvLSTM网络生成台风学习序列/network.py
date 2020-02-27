# -*- coding: utf-8 -*-
##########################导入库#################################
from __future__ import print_function
import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models,layers,optimizers
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,CSVLogger,LambdaCallback
from keras.models import load_model
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,LSTM,Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt


##########################定义模型################################
#ConvLSTM2D
#if return_sequences if data_format='channels_last'
#Input shape:5D tensor with shape(samples, time, rows, cols, channels)
#Output shape:5D tensor with shape(samples, time, output_row, output_col, filters)
def build_model(n_step = 6, row = 500, col = 500, channel = 3, n_hidden_units = 64,output_units = 2):
    print("###############model build###############")
    #input_shape or batch_input_shape in the first layer for automatic build
    # model = models.Sequential()
    # model.add(LSTM(n_hidden_units,
    #         batch_input_shape=(None, n_step, n_input),# (batch,timestep,input_dim)
    #         return_sequences = True,unroll=False, name='LSTM_layer01'))
    # model.add(BatchNormalization())
    # model.add(LSTM(output_units,
    #         return_sequences = True,unroll=False, name='LSTM_layer02'))
    # model.add(BatchNormalization()) # 此步不加，会出现loss不下降
    # model.add(Dense(output_units, name='Dense_layer'))
    # model.add(Activation('softmax', name='activation_layer'))
    print("###############model build###############")
    model = Sequential()
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                        input_shape=(n_step, row, col, channel),
                        padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=3, kernel_size=(3, 3, 3),
                     activation='sigmoid',
                     padding='same', data_format='channels_last'))
    model.summary()
    print ('model.input=',model.input)
    print ('model.input.name=',model.input.name)
    print ('model.input.shape=',model.input.shape)
    print ('model.output=',model.output)
    print ('model.output.name=',model.output.name)
    print ('model.output.shape=',model.output.shape)
    return model
##########################模型训练################################
def train_model(trainX, trainY,validX, validY,model,model_dir = 'model',model_name = "model_best.h5",log_name = "train_log.csv",
                lr = 0.001,epochs = 1000,batch_size = 4) :
    #model_name = "model_best.h5"#"model_Epoch{epoch:04d}-val_mean_absolute_error{val_mean_absolute_error:.2f}.h5"
    print("###############model train###############")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_file = os.path.join(model_dir, model_name)
    #model.compile(loss='binary_crossentropy',
    #			  optimizer='adam',
    #			  metrics=['accuracy'])
    optimizer = optimizers.RMSprop(lr=lr) # Adam RMSprop
    #keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    # model.compile(loss='mean_squared_error', optimizer='RMSprop')
    #model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['mse', 'acc'])
    print('model.metrics_names = ', model.metrics_names) # mean_squared_error mean_absolute_error
    saveBestModel = ModelCheckpoint(model_file, monitor='val_mean_squared_error',verbose=1,
                      save_best_only=False, period = 10)
    earlyStopping = EarlyStopping(monitor='val_mean_squared_error', patience=epochs/10, verbose=1, mode='auto')

    reduce_lr = ReduceLROnPlateau(monitor='val_mean_squared_error',factor=0.1,verbose=1, patience=epochs/100, min_lr=lr/1000.0)
    csv_logger = CSVLogger(os.path.join(model_dir, log_name), separator=',',
                           append=True)  # CSVLogger:将epoch的训练结果保存在csv中
    epoch_end_callback = LambdaCallback(on_epoch_end=lambda epoch, logs:
                                        print("epoch_end_callback epoch", epoch, "lr", logs["lr"])
                                        )  # 自定义的回调函数
    callback_lists = [earlyStopping, saveBestModel, reduce_lr, csv_logger, epoch_end_callback]
    #训练模型。
    history = model.fit(trainX, trainY,
              batch_size=batch_size,
              epochs=epochs,verbose=1, shuffle=False, callbacks = callback_lists,
              validation_data=(validX, validY))
    # 模型保存
    # model.save(model_file)  # elesun
    # print('Model Saved.')  # elesun
    return history
