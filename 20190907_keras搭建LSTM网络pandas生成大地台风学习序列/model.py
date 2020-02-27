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
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,CSVLogger,LambdaCallback
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
            return_sequences = True,unroll=False, name='LSTM_layer01'))
    model.add(BatchNormalization())
    model.add(LSTM(output_units,
            return_sequences = True,unroll=False, name='LSTM_layer02'))
    model.add(BatchNormalization()) # 此步不加，会出现loss不下降
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
def train_model(trainX, trainY,validX, validY,model,model_dir = 'model',model_name = "model_best.h5",log_name = "train_log.csv",
                lr = 0.001,epochs = 10000,batch_size = 8) :
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
    epoch_end_callback = LambdaCallback(on_epoch_end=lambda epoch, logs:
                                        print("epoch_end_callback epoch", epoch, "lr", logs["lr"])
                                        )  # 自定义的回调函数
    callback_lists = [earlyStopping, saveBestModel, reduce_lr, csv_logger, epoch_end_callback]
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
def model_evaluate(model_dir = 'model',model_name = "model_best.h5"):
    print("###############model evaluate###############")
    #加载模型
    print('load the trained model')
    model = load_model(os.path.join(model_dir, model_name))
    print("###############model evaluate###############")
    test_loss,test_metric = model.evaluate(test_data,test_targets,verbose=0)
    print("results test_loss:",test_loss)
    print("results test_mean_absolute_error:",test_metric)

##########################模型预测################################
def model_predict(model,test_data):
    print("###############model predict###############")
    test_input_before_scaler = sc.inverse_transform(test_data[-1:])#把标准化后的一个测试数据再变换回去原始值，作为一条真实输入值
    print("test_input_before_scaler",test_input_before_scaler)
    # the real oral test_data[-1] is  [2.2876e-01 0.0000e+00 8.5600e+00 0.0000e+00 5.2000e-01 6.4050e+00
    #  8.5400e+01 2.7147e+00 5.0000e+00 3.8400e+02 2.0900e+01 7.0800e+01
    #  1.0630e+01]
    test_input_after_scaler = sc.transform(test_input_before_scaler)#进行标准化转换
    print("test_input_after_scaler",test_input_after_scaler)
    # test_input_after_scaler  [[-0.40513088 -0.47409775 -0.4293645  -0.26761547 -0.32876199  0.24715843
    #    0.57957075 -0.49152719 -0.55349978 -0.19321071  1.09799724 -2.96496891
    #   -0.3448435 ]]
    pred_targets = model.predict(test_input_after_scaler, batch_size = 1)
    # the test_targets[-1] is  18.6
    true_targets = test_targets[-1:]
    print("True Target(Price) is ",true_targets)#18.6
    print("Pred Target(Price) is ",pred_targets)
