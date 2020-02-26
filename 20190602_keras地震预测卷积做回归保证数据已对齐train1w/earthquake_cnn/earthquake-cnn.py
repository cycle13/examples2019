# -*- coding: utf-8 -*-
"""
Earthquake time predictor
using keras cnn regress
Author: elesun
"""
from __future__ import print_function
##########################导入库#################################
import numpy as np
import pandas as pd
pd.options.display.precision = 15
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from sklearn.kernel_ridge import KernelRidge
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout
from keras.optimizers import adam,RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
##########################设置参数##############################
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #"1,0"
N_ROWS = 5e6#Number of rows of file to read. Useful for reading pieces of large files.
rows = 150000
data_dir = "../../data/train.csv"
test_dir = "../../data/test/"
model_dir = "./model"
model_name = "model_best.h5"#"model_Epoch{epoch:04d}-val_mean_absolute_error{val_mean_absolute_error:.2f}.h5"
output_dir = "./output"
result_name = "history.png"
submit_name = "submission.csv"
submit_sample_file = "../../data/sample_submission.csv"
epochs = 10000#1000
lr = 0.001
batch_size = 16
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
##########################读取数据################################
print("###############load dataset###############")
train = pd.read_csv(data_dir,#nrows= N_ROWS,
                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
print('load dataset completed !')
X_train = train['acoustic_data'].values
y_train = train['time_to_failure'].values
print("X_train[0:2]\n",X_train[0:2])# 12 6
print("y_train[0:2]\n",y_train[0:2])# 1.4690999832  1.4690999821
del train

# Cut training data
X_train = X_train[:int(np.floor(X_train.shape[0] / rows))*rows]
y_train = y_train[:int(np.floor(y_train.shape[0] / rows))*rows]
X_train= X_train.reshape((-1, rows, 1))#(4194,150000,1)
print("X_train.shape",X_train.shape)
print("X_train[:3,-1,0]",X_train[:3,-1,0])#第0行最后一列（0:原始150001行），第1行最后一列（5:原始300001行），第2行最后一列（3:原始450001行）
y_train = y_train[rows-1::rows]#对行进行操作，间隔rows150000，取第rows-1（含149999）到最后一行（不含150000）的数
print("y_train.shape",y_train.shape)#(4194,1)
print("y_train[:3]",y_train[:3])#对应标签（1.43079719:原始150001行）（1.39149889:原始300001行）（1.35319609:原始450001行）
# 随机拆分训练集与测试集
# Training/ Vaidation Split
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,
                                                  test_size= 0.2,random_state= 11)
##########################数据预处理################################


##########################模型构建################################
def build_model():
    model = Sequential()
    # Conv 1
    model.add(Conv1D(32, 10, activation='relu', input_shape=(rows, 1))) # (rows, 1) 150000
    # Max Pooling
    model.add(MaxPooling1D(100))
    # Conv 3
    model.add(Conv1D(64, 10, activation='relu'))
    # Average Pooling
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    # Output Layer
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    optimizer = RMSprop(lr=lr)
    #keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
    #model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.compile(loss='mse', optimizer=optimizer, metrics=['mean_absolute_error'])#optimizer=adam(lr=lr)
    print('model.metrics_names = ', model.metrics_names)
    model.summary()
    return model
##########################模型训练################################
print("###############model train###############")
model = build_model()
earlyStopping = EarlyStopping(monitor='val_mean_absolute_error',patience=epochs/10,
                              verbose=1,mode='min')
saveBestModel = ModelCheckpoint(os.path.join(model_dir, model_name),save_best_only=True,
                                monitor='val_mean_absolute_error',mode='min',period = 10)
reduce_lr = ReduceLROnPlateau(monitor='val_mean_absolute_error',factor=0.1,
                              patience=epochs/100,verbose=1,
                              mode='min',min_lr=lr/1000.0)
callback_lists = [earlyStopping, saveBestModel, reduce_lr]
history = model.fit(X_train,y_train,
                    batch_size= batch_size,epochs= epochs,
                    validation_data= (X_val, y_val),
                    callbacks=callback_lists,
                    verbose= 2)
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
val_loss,val_metric = model.evaluate(X_val,y_val,verbose=0)
print("results val_loss:",val_loss)
print("results val_mean_absolute_error:",val_metric)

##########################模型预测################################
print("###############model predict###############")
# Submission
submission = pd.read_csv(submit_sample_file, index_col='seg_id')
X_test = []
for segment in tqdm(submission.index):
        seg = pd.read_csv(test_dir + segment + '.csv')
        x = pd.Series(seg['acoustic_data'].values)
        X_test.append(x)
X_test = np.asarray(X_test)
X_test = X_test.reshape((-1, 1))
print("X_test.shape",X_test.shape)
X_test = X_test[:int(np.floor(X_test.shape[0] / rows))*rows]
X_test= X_test.reshape((-1, rows, 1))
print("X_test.shape",X_test.shape)
submission['time_to_failure'] = model.predict(X_test)
submission.to_csv(os.path.join(output_dir, submit_name))
print("saved to ",submit_name)