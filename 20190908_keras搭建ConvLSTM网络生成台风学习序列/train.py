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
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth=True
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
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # "-1" #elesun "1,0"
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
	train_data_dir = "data/train1"
	trainX, trainY, validX, validY = generate.generate_seq(train_data_dir)
	#搭建网络训练模型
	n_step = 6
	row = 500
	col = 500
	channel = 3
	model = network.build_model()
	history = network.train_model(trainX, trainY, validX, validY, model)

	time2 = time.time()
	print('time use:' + str(time2 - time1) + 's')