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
import numpy as np
import pandas as pd
import keras
import time
import datetime

import proprocess
import network

from keras import models, optimizers
from keras.layers import Dense, Dropout
from keras.models import load_model,model_from_json
from keras import backend as K
from sklearn import preprocessing
import datetime
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    time1 = time.time()
    data_path = "dataset/train_sales_data.csv"
    ####################  ######################
    mode = "train"  # train test docker
    print("#################work mode", mode, "#######################")
    if mode == "train":
        # 数据预处理
        #(trainX, trainY) = proprocess.generate_train_seq(train_images_folder_path, train_track_folder_path)
        load_data = proprocess.DataSets
        trainX, trainY, validX, validY = load_data.load_passenger_car(data_path)
        model = network.build_network()
        history = network.train_network(trainX, trainY, validX, validY, model, epochs=1000)
        network.plt_result(history, "output", "history.png")
    elif mode == "test":
        network.helloworld()
    else:
        print("mode error!")
    time2 = time.time()
    print('time use:' + str(time2 - time1) + 's')
