# -*- coding: utf-8 -*-
"""
generate_typhoon_data_for network
batch,frames,high,width,channel
"""
import numpy as np
import os
import cv2
import numpy as np
from PIL import Image
import logging
import random
import tensorflow as tf
import re

# 根据训练数据集生成网络所需要的序列
def generate_seq(train_data_dir,seq_len=12, input_len=6, intervel=6, valid_rate=0.1):
    print('**************generate_seq**************')
    frames_np = []  # 用于保存符合要求可用的文件数据列表
    years = ['A', 'B', 'C', 'D']
    png_file_list = os.listdir(train_data_dir)
    # print('png_file_list is ', png_file_list)
    print('num of png_file_list =', len(png_file_list))
    temp_wai_list = []
    for year in years :
        temp_nei_list = []
        for index_testset, png_filename in enumerate(png_file_list):
            if ("".join(re.findall("^[A-Z]", png_filename))) == year:
                temp_nei_list.append(png_filename)
        temp_nei_list.sort(key=lambda x: int("".join(re.findall("(?<=Hour_).*(?=.png)", x))))
        #print('temp_nei_list is ', temp_nei_list)
        print('num of temp_nei_list',len(temp_nei_list),'for year',year)
        temp_wai_list.append(temp_nei_list)
    #print('temp_wai_list is ', temp_wai_list)
    print('num of temp_wai_list =', len(temp_wai_list))
    typhoon_seq = []
    typhoon_seq_names = []
    temp_nei_list = []
    for i_odd in range(intervel):  # intervel  0 1 2 3 4 5
        #print("i_odd",i_odd)
        for temp_nei_list in temp_wai_list:
            #print('num of temp_nei_list =', len(temp_nei_list))
            index_seq = 0
            frames_np = []
            frames_names = []
            for index_filename, png_filename in enumerate(temp_nei_list):
                year_num = re.findall("^[A-Z]", png_filename)
                year_num = "".join(year_num)
                # print("year_num is", year_num)
                hour_num = re.findall("(?<=Hour_).*(?=.png)", png_filename)
                hour_num = int("".join(hour_num))
                #print("hour_num is", hour_num)
                if hour_num % intervel == i_odd:
                    index_seq = index_seq + 1
                    file_path = os.path.join(train_data_dir, png_filename)
                    #print("%s has been found!" % file_path)
                    # cv2.imread('test.jpg') # BGR
                    image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB) # cv2.COLOR_BGR2GRAY  cv2.COLOR_BGR2RGB elesun channel three
                    # image = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE) #elesun cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE
                    #print('image.shape :', image.shape)  # [501,501,3]
                    image = np.array(image, dtype=np.float32) / 255.0  # elesun
                    # image = image.reshape(image_width, image_width, 1)  # elesun
                    #print('image\n', image)
                    frames_np.append(image)
                    frames_names.append(png_filename)
                    if index_seq == seq_len :#凑够12帧生成一个学习序列
                        #print("len frames_np",len(frames_np))
                        typhoon_seq.append(frames_np)
                        typhoon_seq_names.append(frames_names)
                        frames_np = []
                        frames_names = []
                        index_seq = 0
                        #print("len typhoon_seq_names",len(typhoon_seq_names))
                        #print("len typhoon_seq", len(typhoon_seq))

    print("typhoon_seq_names",typhoon_seq_names)#
    print("num of typhoon_seq_names", len(typhoon_seq_names))  #
    # print("typhoon_seq",typhoon_seq)#
    typhoon_seq = np.array(typhoon_seq) # np.float
    # print("typhoon_seq", typhoon_seq)
    print("type typhoon_seq", type(typhoon_seq))
    print("typhoon_seq.shape", typhoon_seq.shape)  # (num,12,500,500,3)
    train_seq = typhoon_seq[:,:,:,:,:]
    valid_seq = typhoon_seq[::int(1.0/valid_rate),:,:,:,:]
    # print("train_seq\n",train_seq)
    print("train_seq.shape", train_seq.shape)
    print("valid_seq.shape", valid_seq.shape)
    trainX = train_seq[:,:input_len,:,:,:]
    trainY = train_seq[:,(input_len-seq_len):,:,:,:]
    validX = valid_seq[:,:input_len,:,:,:]
    validY = valid_seq[:,(input_len-seq_len):,:,:,:]
    print('trainX.shape:', trainX.shape)
    print('trainY.shape:', trainY.shape)
    print('validX.shape:', validX.shape)
    print('validY.shape:', validY.shape)
    #return train_seq, valid_seq
    return trainX, trainY, validX, validY


if __name__ == '__main__':
    train_data_dir = "data/train1"
    test_data_dir = "data/test"
    trainX, trainY, validX, validY = generate_seq(train_data_dir)
