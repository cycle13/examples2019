#! /usr/bin/env python
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
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,CSVLogger,LambdaCallback,TensorBoard
from keras.models import load_model
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,LSTM,Activation
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

def helloworld():
    print('hello world')

##########################定义模型################################
def build_model(n_step = 12,n_input = 5,n_hidden_units = 64,output_units = 2):
    #n_step = 28, n_input = 28, n_hidden = 128, output_units = 10
    print("###############model build###############")
    #input_shape or batch_input_shape in the first layer for automatic build
    model = models.Sequential()
    model.add(LSTM(n_hidden_units,
            batch_input_shape=(None, n_step, n_input),# (batch,timestep,input_dim)
            return_sequences = True,unroll=False,
            dropout=0,recurrent_dropout=0.5,
            name='LSTM_layer01'))
    # model.add(BatchNormalization())
    model.add(LSTM(n_hidden_units,
                   return_sequences=True, unroll=False,
                   dropout=0, recurrent_dropout=0,
                   name='LSTM_layer02'))
    #model.add(BatchNormalization())
    model.add(LSTM(output_units,
            return_sequences = True,unroll=False,
            dropout=0.5, recurrent_dropout=0,
            name='LSTM_layer03'))
    #model.add(BatchNormalization()) # 此步不加，会出现loss不下降
    # model.add(Dense(output_units, name='Dense_layer'))
    # model.add(Activation('softmax', name='activation_layer'))
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
                lr = 0.001,epochs = 1000,batch_size = 8) :
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
    print('model.metrics_names = ', model.metrics_names) # mean_squared_error mean_absolute_error
    saveBestModel = ModelCheckpoint(model_file, monitor='val_mean_squared_error',verbose=1,
                      save_best_only=False, period = 10)
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
    callback_lists = [earlyStopping, saveBestModel, reduce_lr, csv_logger, tensor_cv, epoch_end_callback]
    #训练模型。
    history = model.fit(trainX, trainY,
              batch_size=batch_size,
              epochs=epochs,verbose=2, shuffle=False, callbacks = callback_lists,
              validation_data=(validX, validY))
    # 模型保存
    # model.save(model_file)  # elesun
    # print('Model Saved.')  # elesun
    return history

##########################展示训练结果################################
def plt_result(history,output_dir = "output",result_name = "history.png"):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print("###############show result###############")
    #print('history.history.keys = ',history.history.keys())
    #print('history.history = ',history.history)
    #print('history.epoch = ',history.epoch)
    # plot history
    plt.title("model performace")
    plt.plot(history.epoch,history.history['loss'], label='train_loss')
    plt.plot(history.epoch,history.history['val_loss'], label='val_loss')
    plt.plot(history.epoch,history.history['mean_squared_error'], label='mean_squared_error')
    plt.plot(history.epoch,history.history['val_mean_squared_error'], label='val_mean_squared_error')

    plt.ylabel("loss or metric")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig(os.path.join(output_dir, result_name))
    print(result_name,"has been saved!")
    #plt.show()

##########################模型评估################################
def model_evaluate(validX, validY, model_dir = 'model', model_name = "model_best.h5"):
    print("###############model evaluate###############")
    #加载模型
    print('load the trained model')
    model = load_model(os.path.join(model_dir, model_name))
    valid_loss,valid_metric = model.evaluate(validX, validY, verbose=0)
    print("results test_loss:",valid_loss)
    print("results test_mean_squared_error:",valid_metric)

##########################模型预测################################
def model_predict(testX,lat_max_val,lat_min_val,lon_max_val,lon_min_val, model_dir = 'model', model_name = "model_best.h5"):
    print("###############model predict###############")
    # 加载模型
    print('load the trained model')
    model = load_model(os.path.join(model_dir, model_name))
    #print("testX\n",testX)
    pred_targets = model.predict(testX)#, batch_size = 1)
    #print("pred_targets\n",pred_targets)
    pred_targets[:,:,0] = pred_targets[:,:,0]*(lat_max_val - lat_min_val) + lat_min_val #反归一化
    pred_targets[:,:,1] = pred_targets[:,:,1]*(lon_max_val - lon_min_val) + lon_min_val
    #print("pred_targets after back_norm\n", pred_targets)
    return pred_targets
