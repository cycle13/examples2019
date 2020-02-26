# -*- coding: utf-8 -*-
from __future__ import print_function
##########################导入库#################################
import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import models,layers,optimizers
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.models import load_model
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D
import matplotlib.pyplot as plt
##########################设置超参数##############################
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # -1 1 0
data_dir = "./boston_housing.npz"
model_dir = "./model"
model_name = "model_best.h5"#"model_Epoch{epoch:04d}-val_mean_absolute_error{val_mean_absolute_error:.2f}.h5"
output_dir = "./output"
result_name = "history.png"
epochs = 1000#1000
lr = 0.001
batchsize = 16

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
##########################读取数据################################
print("###############load dataset###############")
#(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()
with np.load(data_dir) as f:
    x = f['x']
    y = f['y']
# 随机拆分训练集与测试集
train_data, test_data, train_targets, test_targets = train_test_split(x, y, test_size=0.2, random_state=10)
print('the type of train data is ',train_data.dtype)
print('the shape of train data is ',train_data.shape)
print('the shape of train target is ',train_targets.shape)
print('the type of train target is ',train_targets.dtype)
print('train target[:10]:',train_targets[:10])
print('the shape of test data is ',test_data.shape)
print('the shape of test target is ',test_targets.shape)
print('the test_data[-1] is ',test_data[-1])
print('the test_targets[-1] is ',test_targets[-1])
##########################数据预处理################################
#一种常见的数据处理方法是特征归一化normalization—减均值除以标准差；数据0中心化，方差为1.
print("###############data proecess###############")
# mean = train_data.mean(axis=0)
# print('the shape of mean is ',mean.shape)
# print('train_data mean = ',mean)
# train_data -= mean # 减去均值
# std = train_data.std(axis=0) # 特征标准差
# print('the shape of std is ',std.shape)
# print('train_data std = ',std)
# train_data /= std
# test_data -= mean #测试集处理：使用训练集的均值和标准差；不用重新计算
# test_data /= std
# 为了追求机器学习和最优化算法的最佳性能，我们将特征缩放
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(train_data) # 估算每个特征的平均值和标准差
# 查看特征的平均值
print('the shape of mean is ',sc.mean_.shape)
print('train_data mean = ',sc.mean_)
# 查看特征的标准差
print('the shape of std is ',sc.scale_.shape)
print('train_data std = ',sc.scale_)
train_data = sc.transform(train_data)
# 注意：这里我们要用同样的参数来标准化测试集，使得测试集和训练集之间有可比性
test_data = sc.transform(test_data)
##########################模型构建################################
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu',input_shape=(train_data.shape[1],)))
    print("train_data.shape[1] = ",train_data.shape[1])
    model.add(Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(1))
    optimizer = optimizers.RMSprop(lr=lr)
    #keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    print('model.metrics_names = ', model.metrics_names)
    model.summary()
    return model

##########################模型训练################################
print("###############model train###############")
model = build_model()
saveBestModel = ModelCheckpoint(os.path.join(model_dir, model_name), monitor='val_mean_absolute_error',verbose=1,
                            save_best_only=False, period = 10)
earlyStopping = EarlyStopping(monitor='val_mean_absolute_error', patience=epochs/10, verbose=1, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_mean_absolute_error',factor=0.1,verbose=1, patience=epochs/100, min_lr=lr/1000.0)

callback_lists = [earlyStopping, saveBestModel, reduce_lr]
history = model.fit(train_data, train_targets,epochs=epochs, batch_size=batchsize,
                    validation_split=0.2, verbose=2, shuffle=True, callbacks = callback_lists)
# 模型保存
# model.save(model_file)  # elesun
# print('Model Saved.')  # elesun
##########################展示训练结果################################
print('history.history.keys = ',history.history.keys())
print('history.history = ',history.history)
print('history.epoch = ',history.epoch)
# plot history
plt.title("model performace")
plt.plot(history.epoch,history.history['loss'], label='train_loss')
plt.plot(history.epoch,history.history['val_loss'], label='val_loss')
plt.plot(history.epoch,history.history['mean_absolute_error'], label='mean_absolute_error')
plt.plot(history.epoch,history.history['val_mean_absolute_error'], label='val_mean_absolute_error')

plt.ylabel("loss or metric")
plt.xlabel("epochs")
plt.legend()
plt.savefig(os.path.join(output_dir, result_name))
plt.show()
##########################模型评估################################
#加载模型
print('load the trained model')
model = load_model(os.path.join(model_dir, model_name))
print("###############model evaluate###############")
test_loss,test_metric = model.evaluate(test_data,test_targets,verbose=0)
print("results test_loss:",test_loss)
print("results test_mean_absolute_error:",test_metric)

##########################模型预测################################
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