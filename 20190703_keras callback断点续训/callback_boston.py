# -*- coding: utf-8 -*-
from __future__ import print_function
##########################导入库#################################
import os
import sys
import time
import numpy as np
import pandas as pd
from keras.datasets import boston_housing
import tensorflow as tf
from keras import models,layers,optimizers
from keras.callbacks import ModelCheckpoint,CSVLogger,ReduceLROnPlateau,LambdaCallback
from keras.models import load_model
import matplotlib.pyplot as plt
##########################设置超参数##############################
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # -1 1 0
model_dir = "./model"
model_name = "model_Epoch{epoch:04d}_val_mean_absolute_error{val_mean_absolute_error:.3f}.h5"
log_name = "train_log.csv"
output_dir = "./output"
result_name = "history.png"
epochs = 10000#1000
initial_epoch = 0
lr = 0.001
batchsize = 16

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
##########################读取数据################################
print("###############load dataset###############")
(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()
print('the type of train data is ',train_data.dtype)
print('the shape of train data is ',train_data.shape)
print('the shape of train target is ',train_targets.shape)
print('the type of train target is ',train_targets.dtype)
print('train target[:10]:',train_targets[:10])
print('the shape of test data is ',test_data.shape)
print('the shape of test target is ',test_targets.shape)

##########################数据预处理################################
#一种常见的数据处理方法是特征归一化normalization—减均值除以标准差；数据0中心化，方差为1.
print("###############data proecess###############")
mean = train_data.mean(axis=0)
print('the shape of mean is ',mean.shape)
train_data -= mean # 减去均值
std = train_data.std(axis=0) # 特征标准差
print('the shape of std is ',std.shape)
train_data /= std
test_data -= mean #测试集处理：使用训练集的均值和标准差；不用重新计算
test_data /= std
##########################模型构建################################
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.summary()
    return model

##########################模型训练################################
print("###############model train###############")
print("building network ...")
model = build_model()
if os.path.exists(model_dir):
    print("loading model ...")
    filelists = os.listdir(model_dir)
    modellists = []
    for filename in filelists :
        if filename[-2:] == "h5" :
            modellists.append(filename)
    print('modellists length ', len(modellists))
    if len(modellists) != 0 : #没有模型文件存在
        modellists.sort(key=lambda x:int(x[11:15])) #model_Epoch0220_val_mean_absolute_error2.366.h5
        #print('modellists\n', modellists)
        last_model_name = modellists[len(modellists)-1]
        initial_epoch = int(last_model_name[11:15])
        if os.path.isfile(os.path.join(model_dir, log_name)):
            df = pd.read_csv(os.path.join(model_dir, log_name),
                             index_col="epoch", header = 0, dtype={"lr":np.float32})
            print("df[0:5]\n",df[0:5])
            lr = df['lr'].values[initial_epoch]
            print("lr",lr,"at",initial_epoch,"from last trained log file.")
        model = load_model(os.path.join(model_dir, last_model_name))
        # 若成功加载前面保存的参数，输出下列信息
        print("load model success !")

optimizer = optimizers.RMSprop(lr=lr)
# keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
print('model.metrics_names = ', model.metrics_names)
checkpoint = ModelCheckpoint(os.path.join(model_dir, model_name), monitor='val_mean_absolute_error',verbose=1,
                            save_best_only=False, period = 10)#save_best_only=False每间隔period个epoch都要保存模型，不管模型好坏
reduce_lr = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.5,#new_lr=factor*old_lr
                              patience=epochs/10, min_lr=lr/100)#有了此callback就有了返回值lr
csv_logger = CSVLogger(os.path.join(model_dir, log_name), separator=',', append=True) #CSVLogger:将epoch的训练结果保存在csv中
epoch_end_callback = LambdaCallback(on_epoch_end=lambda epoch, logs:
                                    print("epoch_end_callback epoch",epoch,"lr",logs["lr"])
                                    )#自定义的回调函数
callback_lists = [checkpoint, reduce_lr, csv_logger, epoch_end_callback]
history = model.fit(train_data, train_targets,epochs=epochs, batch_size=batchsize,
                    validation_split=0.2, verbose=2, shuffle=True,
                    initial_epoch = initial_epoch, callbacks = callback_lists)
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
print("###############model evaluate###############")
test_loss,test_metric = model.evaluate(test_data,test_targets,verbose=0)
print("results test_loss:",test_loss)
print("results test_mean_absolute_error:",test_metric)

##########################模型预测################################
print("###############model predict###############")