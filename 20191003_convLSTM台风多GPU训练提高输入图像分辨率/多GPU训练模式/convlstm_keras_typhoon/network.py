# -*- coding: utf-8 -*-
##########################导入库#################################
from __future__ import print_function
import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models,layers,optimizers
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,CSVLogger,LambdaCallback,TensorBoard
from keras.models import load_model
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,LSTM,Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.utils.training_utils import multi_gpu_model   #导入keras多GPU函数

##########################定义模型################################
#ConvLSTM2D
#if return_sequences if data_format='channels_last'
#Input shape:5D tensor with shape(samples, time, rows, cols, channels)
#Output shape:5D tensor with shape(samples, time, output_row, output_col, filters)
def build_model(n_step = 6, row = 200, col = 200, channel = 3, n_filters = 32,output_filters = 3):
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
    model = Sequential()
    model.add(ConvLSTM2D(filters=n_filters, kernel_size=(3, 3),
                        input_shape=(n_step, row, col, channel),
                        padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=n_filters, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=n_filters, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=n_filters, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=output_filters, kernel_size=(3, 3, 3),
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
def train_model(trainX, trainY,validX, validY,model,model_dir = 'model',model_name = "model_best.h5",log_name = "train_log.csv",tensor_log_name = "tensorboard_logs",
                lr = 0.001,epochs = 100,batch_size = 4) :
    #model_name = "model_best.h5"#"model_Epoch{epoch:04d}-val_mean_absolute_error{val_mean_absolute_error:.2f}.h5"
    print("###############model train###############")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_file = os.path.join(model_dir, model_name)
    mgpus_model = multi_gpu_model(model, gpus=2)  #elesun gpus 多GPU并行训练 设置使用2个gpu，该句放在模型compile之前
    optimizer = optimizers.RMSprop(lr=lr, clipvalue = 50) #elesun 梯度裁剪 Adam  RMSprop适合于RNN网络训练
    #keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)

    #model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    mgpus_model.compile(optimizer=optimizer, loss='mse', metrics=['mse']) #elesun gpus
    print('model.metrics_names = ', mgpus_model.metrics_names) # mean_squared_error mean_absolute_error
    saveBestModel = ModelCheckpoint(model_file, monitor='val_mean_squared_error',verbose=1,
                      save_best_only=True, period = 1)
    earlyStopping = EarlyStopping(monitor='val_mean_squared_error', patience=epochs/10, verbose=1, mode='auto')

    reduce_lr = ReduceLROnPlateau(monitor='val_mean_squared_error',factor=0.1,verbose=1, patience=epochs/100, min_lr=lr/1000.0)
    csv_logger = CSVLogger(os.path.join(model_dir, log_name), separator=',',
                           append=True)  # CSVLogger:将epoch的训练结果保存在csv中
    tensor_cv = TensorBoard(log_dir=os.path.join(model_dir, tensor_log_name),  # log 目录
                             histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                             batch_size=32,  # 用多大量的数据计算直方图
                             write_graph=True,  # 是否存储网络结构图
                             write_grads=True,  # 是否可视化梯度直方图
                             write_images=True,  # 是否可视化参数
                             embeddings_freq=0,
                             embeddings_layer_names=None,
                             embeddings_metadata=None)
    epoch_end_callback = LambdaCallback(on_epoch_end=lambda epoch, logs:
                                        print("epoch_end_callback epoch", epoch, "lr", logs["lr"])
                                        )  # 自定义的回调函数
    callback_lists = [earlyStopping, reduce_lr, csv_logger, tensor_cv, epoch_end_callback] #elesun gpus del saveBestModel earlyStopping, saveBestModel, reduce_lr,

    #训练模型。
    #history = model.fit(trainX, trainY,  # elesun
    history = mgpus_model.fit(trainX, trainY,# elesun gpus
              batch_size=batch_size,
              epochs=epochs,verbose=2, shuffle=True, callbacks = callback_lists,
              validation_data=(validX, validY))
    # 模型保存
    model.save("model_trainend.h5")  # elesun
    model.save_weights("model_weights_trainend.h5")  # elesun
    print('Model Saved after train end')  # elesun
    return history
