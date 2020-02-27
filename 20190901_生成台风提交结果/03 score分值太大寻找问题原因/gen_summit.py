# -*- coding: utf-8 -*-
"""
from inputimgs to generate outputimgs and transform to npy, then generate summit.tar.gz
for typhoon test
env:
	Linux ubuntu 4.4.0-31-generic x86_64 GNU;python 2.7;tensorflow1.10.1;Keras2.2.4
	matplotlib2.2.3
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
import re
import cv2

def generate_images(input_dir,mid_dir,output_dir) :
    if tf.gfile.Exists(mid_dir):
        tf.gfile.DeleteRecursively(mid_dir)
    tf.gfile.MakeDirs(mid_dir)
    if tf.gfile.Exists(output_dir):
        tf.gfile.DeleteRecursively(output_dir)
    tf.gfile.MakeDirs(output_dir)
    # if not os.path.exists(mid_dir):
    #     os.makedirs(mid_dir)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    print('load data from dir :', input_dir)
    file_list = os.listdir(input_dir)
    for year in years:
        year_file_list = []
        year_file_valid_list = []
        for index_file, file_name in enumerate(file_list):
            if ("".join(re.findall("^[A-Z]", file_name))) == year:
                year_file_list.append(file_name)
        year_file_list.sort(key=lambda x: int("".join(re.findall("(?<=Hour_).*(?=.png)", x))))
        #print("year_file_list",year_file_list)
        for i in range((input_len - 1), -1, -1):  # 向前回溯self.seq_len-input
            year_file_valid_list.append(year_file_list[-1 - (i * interval)])  # 每步移动interval
        print("year_file_valid_list", year_file_valid_list)
        print("len(year_file_valid_list)",len(year_file_valid_list))
        # ['U_Hour_186.png', 'U_Hour_192.png', 'U_Hour_198.png', 'U_Hour_204.png', 'U_Hour_210.png','U_Hour_216.png']
        for filename in year_file_valid_list:
            path = ''
            path = input_dir + '/' + filename
            ####################读入图像###############################
            image = cv2.imread(path, cv2.IMREAD_COLOR)  #方式读取图片 cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE
            ####################写入图像########################
            path = mid_dir + '/' + filename
            cv2.imwrite(path, image)
            print("%s has been saved!" % path)
        for i in range(1,predict_step+1,1):  # 向后拷贝self.seq_len-input
            filename = year_file_valid_list[-1][0:7] + \
                       str(int("".join(re.findall("(?<=Hour_).*(?=.png)", year_file_valid_list[-1]))) # 每步移动interval
                           + (i * interval)) + ".png"
            ####################写入图像########################
            path = mid_dir + '/' + filename
            cv2.imwrite(path, image)
            print("%s has been saved!" % path)

            # void resize(InputArray src, OutputArray dst, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR );
            res = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)  #
            # INTER_NEAREST - 最邻近插值
            # INTER_LINEAR - 双线性插值，如果最后一个参数你不指定，默认使用这种方法
            # INTER_AREA - resampling using pixel area relation.
            # INTER_CUBIC - 4x4像素邻域内的双立方插值
            # INTER_LANCZOS4 - 8x8像素邻域内的Lanczos插值
            ####################写入图像########################
            path = output_dir + '/' + filename
            cv2.imwrite(path, res)
            print("%s has been resized and saved!" % path)

def generate_summit(output_dir,summit_dir) :
    if tf.gfile.Exists(summit_dir):
        tf.gfile.DeleteRecursively(summit_dir)
    tf.gfile.MakeDirs(summit_dir)
    # os.system("rm summit.tar.gz")
    # print("command rm")
    print('load images from dir :', output_dir)
    file_list = os.listdir(output_dir)
    for index_file, file_name in enumerate(file_list):
        path = ''
        path = output_dir + '/' + file_name
        ####################读入图像###############################
        image = cv2.imread(path, cv2.IMREAD_COLOR)  # 方式读取图片 cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE
        #print("image",image)
        #print("image.shape", image.shape)
        img_np = np.array(image / 255.0 * 4095.0, dtype=np.uint16)
        output_ban08 = img_np[:, :, 0]
        output_ban09 = img_np[:, :, 1]
        output_ban10 = img_np[:, :, 2]
        #filename文件名格式：W_Hour_1264.png
        #npy文件名格式：U_Hour_252_Band_10.npy
        np.save(os.path.join(summit_dir, file_name[:-4] + '_Band_08.npy'), output_ban08)
        np.save(os.path.join(summit_dir, file_name[:-4] + '_Band_09.npy'), output_ban09)
        np.save(os.path.join(summit_dir, file_name[:-4] + '_Band_10.npy'), output_ban10)
        print("%s has been saved!" %(file_name[:-4]+".npy"))
    # time.sleep(5)
    # os.system("cd %s"%summit_dir)
    # print("command cd")
    # os.system("tar -cvzf ../summit.tar.gz *")
    # print("command tar")
    # os.system("cd ..")
    # print("command cd")
    # os.system("sz summit.tar.gz")
    # print("command sz")

def generate_images_fr_genimages(gen_dir, output_dir):
    if tf.gfile.Exists(output_dir):
        tf.gfile.DeleteRecursively(output_dir)
    tf.gfile.MakeDirs(output_dir)
    print('load images from dir :', gen_dir)
    file_list = os.listdir(gen_dir)
    for file_name in file_list:
        path = ''
        path = gen_dir + '/' + file_name
        img_list = os.listdir(path)
        for img_name in img_list:
            if img_name[0:6] == "predic" :
                print("img_name",img_name)
                valid_path = ''
                valid_path = path + '/' + img_name
                print("valid_path",valid_path)
                ####################读入图像###############################
                image = cv2.imread(valid_path, cv2.IMREAD_COLOR)  # 方式读取图片 cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE
                # void resize(InputArray src, OutputArray dst, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR );
                res = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)  #
                # INTER_NEAREST - 最邻近插值
                # INTER_LINEAR - 双线性插值，如果最后一个参数你不指定，默认使用这种方法
                # INTER_AREA - resampling using pixel area relation.
                # INTER_CUBIC - 4x4像素邻域内的双立方插值
                # INTER_LANCZOS4 - 8x8像素邻域内的Lanczos插值
                ####################写入图像########################
                save_path = output_dir + '/' + img_name[7:]
                cv2.imwrite(save_path, res)
                print("%s has been resized and saved!" % save_path)

if __name__ == '__main__':
    input_dir = "input_imgs"
    mid_dir = "mid_imgs"
    output_dir = "output_imgs"
    summit_dir = "output_npy"
    gen_dir = "gen_imgs"
    years = ['U', 'V', 'W', 'X', 'Y', 'Z']
    seq_len = 12
    input_len = 6
    predict_step  = seq_len - input_len
    interval = 6
    width = height = 1999
    time1 = time.time()
    #generate_images(input_dir,mid_dir,output_dir)
    generate_images_fr_genimages(gen_dir, output_dir)
    generate_summit(output_dir,summit_dir)
    time2 = time.time()
    print('time use:' + str(time2 - time1) + 's')

