# -*- coding: utf-8 -*-
"""
titanic servive class prediction
using keras bp 
Author: elesun
"""
from __future__ import print_function
##########################导入库#################################
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import adam,RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras.models import load_model
##########################设置参数##############################
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #"1,0"
data_dir = "./data/titanic3.xls"
test_dir = "./data/test/"
model_dir = "./model"
model_name = "model_best.h5"
output_dir = "./output"
result_name = "history.png"
processed_name = "processed_titanic.csv"
epochs = 10000# 300000
lr = 0.001
batch_size = 32
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
##########################读取数据################################
print("###############load dataset###############")
if not os.path.isfile(data_dir):
    print("download please!")
all_df = pd.read_excel(data_dir)
cols = ['survived','name','pclass','sex','age','sibsp','parch','fare','embarked']
all_df = all_df[cols]
##########################数据预处理################################
# delete name
df = all_df.drop(['name'],axis=1)
# check null
df.isnull().sum()
age_mean = df['age'].mean()
df['age'] = df['age'].fillna(age_mean)
fare_mean = df['fare'].mean()
df['fare'] = df['fare'].fillna(fare_mean)
# sex code
df['sex'] = df['sex'].map({'female':0,'male':1}).astype(int)
#embarked code
df = pd.get_dummies(data=df,columns=['embarked'])
#save to processed_name
df.to_csv(os.path.join(output_dir, processed_name))
print("saved to ",processed_name)
#to array
ndarray = df.values
label = ndarray[:,0]
data = ndarray[:,1:]
#normalized
minmax_scale = preprocessing.MinMaxScaler(feature_range = (0,1))
scaleddate = minmax_scale.fit_transform(data)
# 随机拆分训练集与测试集
# Training/ Vaidation Split
X_train, X_test, y_train, y_test = train_test_split(scaleddate,label,
                                                  test_size= 0.2,random_state= 11)
##########################模型构建################################
#build a model
def build_model():
    model = Sequential()
    model.add(Dense(units=40,input_dim = 9,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(30,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    optimizer = RMSprop(lr=lr)#optimizer = adam(lr=lr)
    #model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])#
    print('model.metrics_names = ', model.metrics_names)
    model.summary()
    return model
##########################模型训练################################
print("###############model train###############")
model = build_model()
earlyStopping = EarlyStopping(monitor='val_acc',patience=epochs/10,
                              verbose=1,mode='max')
saveBestModel = ModelCheckpoint(os.path.join(model_dir, model_name),save_best_only=True,
                                monitor='val_acc',mode='max',period = 10)
reduce_lr = ReduceLROnPlateau(monitor='val_acc',factor=0.1,
                              patience=epochs/100,verbose=1,
                              mode='max',min_lr=lr/1000.0)
callback_lists = [earlyStopping, saveBestModel, reduce_lr]
history = model.fit(X_train,y_train,
                    batch_size= batch_size,epochs= epochs,
                    validation_split = 0.2,
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
plt.plot(history.epoch,history.history['acc'], label='acc')
plt.plot(history.epoch,history.history['val_acc'], label='val_acc')
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
test_loss,test_metric = model.evaluate(X_test,y_test,verbose=0)
print("results val_loss:",test_loss)
print("results val_acc:",test_metric)
