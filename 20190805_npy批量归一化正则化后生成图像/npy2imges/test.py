# -*- coding: utf-8 -*-
""" 
subplots_matplot_test01
fun:
	draw multi-pictures 
env:
	Linux ubuntu 4.4.0-31-generic x86_64 GNU;python 2.7;tensorflow1.10.1;Keras2.2.4
	pip2,matplotlib2.2.3
"""
from __future__ import print_function
import os
import sys
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import scipy.misc

def gen_results(in_dir,out_dir):
    min_val_band08 = 2600.0
    max_val_band08 = 4000.0
    min_val_band09 = 900.0
    max_val_band09 = 4000.0
    min_val_band10 = 2100.0
    max_val_band10 = 3800.0
    if not os.path.exists(in_dir):
        print("there is not directory", in_dir)
    print("input directory is ", in_dir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    testset_file_list = os.listdir(in_dir)
    print('testset_file_list is ', testset_file_list) # 'X_Hour_1520_Band_09.npy', 'Z_Hour_2140_Band_08.npy'
    print('num of testset_file_list =', len(testset_file_list))
    #testset_file_list.sort(key=lambda x: int(x[0:2]))  #
    for index_testset, testset_filename in enumerate(testset_file_list):
        file_path = ''
        file_path = in_dir + '/' + testset_filename
        seqence_num = re.findall("^[A-Z]", testset_filename)
        seqence_num = "".join(seqence_num)
        print("seqence_num is", seqence_num)
        hour_num = re.findall("(?<=Hour_).*(?=_Band)", testset_filename)
        hour_num = int("".join(hour_num))
        print("hour_num is", hour_num)
        band_num = re.findall("(?<=Band_).*(?=.npy)", testset_filename)
        band_num = int("".join(band_num))
        print("band_num is", band_num)
        print("%s has been found!"%file_path)
        matrix = np.load(file_path)
        print("matrix\n",matrix)
        print("matrix.shape",matrix.shape)
        max_val = np.max(matrix)
        min_val = np.min(matrix)
        print("band_num",band_num,"max_val", max_val)  #round1_train_A_part1 08(2600-4000)
        print("band_num",band_num,"min_val", min_val)  #09(900-4000) 10(2100-3800)
        if band_num == 8 :
            matrix = np.maximum(matrix, min_val_band08)  # min_val
            matrix = np.minimum(matrix, max_val_band08)  # max_val
            matrix_scale = (max_val_band08 - matrix) / (max_val_band08 - min_val_band08)
        elif band_num == 9 :
            matrix = np.maximum(matrix, min_val_band09)  # min_val
            matrix = np.minimum(matrix, max_val_band09)  # max_val
            matrix_scale = (max_val_band09 - matrix) / (max_val_band09 - min_val_band09)
        elif band_num == 10 :
            matrix = np.maximum(matrix, min_val_band10)  # min_val
            matrix = np.minimum(matrix, max_val_band10)  # max_val
            matrix_scale = (max_val_band10 - matrix) / (max_val_band10 - min_val_band10)
        else :
            print("no such band num",band_num)
        print("matrix_scale\n", matrix_scale)
        max_val = np.max(matrix_scale)
        min_val = np.min(matrix_scale)
        print("after scale max_val", max_val)
        print("after scale min_val", min_val)
        scipy.misc.imsave(out_dir+"/"+testset_filename[:-4]+".png", matrix_scale)
        print("image has been saved!")
        # new_fliename = re.sub("(?<=Hour_).*(?=_Band)", str(str_filename+6*i), testset_filename)
        # #re.sub(pattern, repl, string, count=0, flags=0)
        # print("new_fliename", new_fliename)
        # np.save(os.path.join(out_dir,new_fliename),matrix)


if __name__ == '__main__':
    in_dir = 'data/round1_train_A_part1'
    out_dir = "out"
    time1 = time.time()
    gen_results(in_dir,out_dir)
    time2=time.time()
    print ('time use: ' + str(time2 - time1) + ' s')