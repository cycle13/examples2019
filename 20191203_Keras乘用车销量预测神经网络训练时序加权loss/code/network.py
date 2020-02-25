# -*- coding: utf-8 -*-
"""
network
fun:

env:
	Linux ubuntu 4.4.0-31-generic x86_64 GNU;python 2.7;tensorflow1.10.1;Keras2.2.4
	pip2,matplotlib2.2.3
"""
from __future__ import print_function
import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import backend as K
from keras import models,layers,optimizers
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,CSVLogger,LambdaCallback
from keras.models import load_model
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,LSTM,Activation
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

def helloworld():
    print('hello world')


##########################定义模型################################
def build_network():
    # 构建模型
    model = models.Sequential()
    model.add(Dense(units=36, input_shape=(12,), activation='relu', kernel_initializer=keras.initializers.glorot_uniform(seed=0),bias_initializer=keras.initializers.Zeros(),name="hidden01"))   #yasong keras.initializers.glorot_uniform(seed=0)  kernel_initializer=keras.initializers.Ones()
    #model.add(BatchNormalization())
    #model.add(Dense(units=12, activation='relu',kernel_initializer=keras.initializers.glorot_uniform(seed=0),bias_initializer=keras.initializers.Zeros(),name="hidden02"))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Dense(units=4, activation='relu',kernel_initializer=keras.initializers.glorot_uniform(seed=0),bias_initializer=keras.initializers.Zeros(), name="output"))
    #model.add(BatchNormalization())
    model.summary()
    return model

def user_loss(y_true, y_pred):
    #print("y_true",y_true.shape)
    # loss_val = (K.mean(K.square(y_pred - y_true), axis=-1))
    loss_val = (0.5*(K.square(y_pred[:,1] - y_true[:,0])) +
                0.25*(K.square(y_pred[:,1] - y_true[:,1])) +
                0.15*(K.square(y_pred[:,2] - y_true[:,2])) +
                0.1*(K.square(y_pred[:,3] - y_true[:,3])))
    return loss_val

##########################模型训练################################
def train_network(trainX, trainY, validX, validY,
                  model, model_dir='model', model_name="model_Epoch{epoch:04d}-val_mean_squared_error{val_mean_squared_error:.2f}.h5",
                  log_name="train_log.csv",
                  lr=0.01, epochs=1000, batch_size=8,loss_fun=user_loss):
    # model_name = "model_best.h5"#"model_Epoch{epoch:04d}-val_mean_absolute_error{val_mean_absolute_error:.2f}.h5"
    print("###############model train###############")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_file = os.path.join(model_dir, model_name)
    optimizer = optimizers.Adam(lr=lr)  # Adam RMSprop SGD
    model.compile(optimizer=optimizer, loss=loss_fun, metrics=['mse'])
    print('model.metrics_names = ', model.metrics_names)  # mean_squared_error mean_absolute_error
    saveBestModel = ModelCheckpoint(model_file, monitor='val_mean_squared_error', verbose=1,
                                    save_best_only=False, period=10)
    earlyStopping = EarlyStopping(monitor='val_mean_squared_error', patience=epochs / 10, verbose=1, mode='auto')

    reduce_lr = ReduceLROnPlateau(monitor='val_mean_squared_error', factor=0.1, verbose=1, patience=epochs / 100,
                                  min_lr=lr / 1000.0)
    csv_logger = CSVLogger(os.path.join(model_dir, log_name), separator=',',
                           append=True)  # CSVLogger:将epoch的训练结果保存在csv中
    epoch_end_callback = LambdaCallback(on_epoch_end=lambda epoch, logs:
    print("epoch_end_callback epoch", epoch, "lr", logs["lr"])
                                        )  # 自定义的回调函数
    callback_lists = [earlyStopping, saveBestModel, reduce_lr, csv_logger, epoch_end_callback]
    # 训练模型。
    history = model.fit(trainX, trainY,
                        batch_size=batch_size,
                        epochs=epochs, verbose=2,
                        shuffle=False,
                        #callbacks=callback_lists,
                        validation_data=(validX, validY)
                        )
    # 模型保存
    now = datetime.datetime.now()
    print("now date time:",now)
    #print(history.history['mean_squared_error'][-1])
    model_name = "%04d"%(now.year)+"%02d"%(now.month)+"%02d"%(now.day)+"%02d"%(now.hour)+"%02d"%(now.minute) + \
                 "_epoch{epoch:04d}-val_mean_squared_error{val_mean_squared_error:.2f}.h5".format(epoch=history.epoch[-1],val_mean_squared_error=history.history['val_mean_squared_error'][-1])
    model_file = os.path.join(model_dir, model_name)
    model.save(model_file)  # elesun
    print(model_file,'Model Saved.')  # elesun
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
    #plt.plot(history.epoch,history.history['val_loss'], label='val_loss')
    #plt.plot(history.epoch,history.history['mean_squared_error'], label='mean_squared_error')
    #plt.plot(history.epoch,history.history['val_mean_squared_error'], label='val_mean_squared_error')

    plt.ylabel("loss or metric")
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig(os.path.join(output_dir, result_name))
    print(result_name,"has been saved!")
    plt.show()
