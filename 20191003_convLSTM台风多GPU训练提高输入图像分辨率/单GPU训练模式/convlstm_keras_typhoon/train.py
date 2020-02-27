# -*- coding: utf-8 -*-
"""
typhoon_train
env:
	Linux ubuntu 4.4.0-31-generic x86_64 GNU;python 2.7;tensorflow1.10.1;Keras2.2.4
	matplotlib2.2.3
structure:
	datasets --source datasets npy
	data -- after process like png
	model -- saved model
	output -- output constrast img
	generate -- data to network
	nework -- model build
	train -- train and model save
"""
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import psutil
################gpu config##################################
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.85 # 配置每个 GPU 上占用的内存的比例
#per_process_gpu_memory_fraction指定了每个GPU进程中使用显存的上限，但它只能均匀作用于所有GPU，无法对不同GPU设置不同的上限
config.gpu_options.allow_growth = True # True时，分配器将不会指定所有的GPU内存，而是根据需求增长
sess = tf.Session(config=config)
KTF.set_session(sess)
############################################################
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model

import network
import process
import generate

if __name__ == '__main__':
	time1 = time.time()
	seq_len = 12
	input_len = 6
	interval = 6
	#数据预处理
	# process_dir = '/home/data/sfz/data_typhoon/npy_test_0923/'  # dataset/npy_testset/
	# output_dir = '/home/data/sfz/data_typhoon/png_test_0923/'  # ./img_data/test/
	# HEADS = ['U', 'V', 'W', 'X', 'Y', 'Z']
	# print('Searching for missing data of train dataset...')
	# filling(process_dir, HEADS)
	# print('Transfering...')
	# save2img(process_dir, output_dir, HEADS, 500)
	# print('Save images OK')
	# process_dir = '/home/data/sfz/data_typhoon/npy_trainset_all0923/'  # dataset/npy_trainset/
	# output_dir = '/home/data/sfz/data_typhoon/png_train_all0923/'  # img_data/train/
	# HEADS = ['A', 'B', 'C', 'D']
	# print('Searching for missing data of test dataset...')
	# filling(process_dir, HEADS)
	# print('Transfering...')
	# save2img(process_dir, output_dir, HEADS, 500)
	# print('Save images OK')
	#训练序列生成
	n_step = 6
	row = 55 # 100
	col = 55 # 100
	channel = 3
	#train_data_dir = "/home/data/sfz/data_typhoon/png_train_small"#"data/train1" png_train_small
	#train_data_dir = "/home/data/sfz/data_typhoon/png_train_all0923"
	train_data_dir = "/home/data/sfz/data_typhoon/png_train_allA"
	trainX, trainY, validX, validY = generate.generate_seq(train_data_dir,row = row, col = col)
	#搭建网络训练模型
	model = network.build_model(n_step = n_step, row = row, col = col, channel = channel)
	history = network.train_model(trainX, trainY, validX, validY, model)
	time2 = time.time()
	print('time use:' + str(time2 - time1) + 's')