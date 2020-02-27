# -*- coding: utf-8 -*-
""" 

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

def gen_results(in_dir):
    if not os.path.exists(in_dir):
        print("there is not directory", in_dir)
    print("input directory is ", in_dir)

    testset_file_list = os.listdir(in_dir)
    print('testset_file_list is ', testset_file_list) # 'X_Hour_1520_Band_09.npy', 'Z_Hour_2140_Band_08.npy'
    print('num of testset_file_list =', len(testset_file_list))
    #testset_file_list.sort(key=lambda x: int(x[0:2]))  #
    for index_testset, testset_filename in enumerate(testset_file_list):
        file_path = ''
        file_path = in_dir + '/' + testset_filename
        #str_filename = re.findall("(?<=Hour_).*(?=_Band)", testset_filename)
        #str_filename = int("".join(str_filename))
        #print("str_filename is", str_filename)
        print("testset_filename is", testset_filename)
        matrix = np.load(file_path)
        print("matrix\n",matrix)
        print("matrix.shape",matrix.shape)


if __name__ == '__main__':
    #in_dir = 'output_npy'
    in_dir = 'npy/output_npy_score471'
    #in_dir = 'npy/output_npy_score3W'
    #in_dir = 'npy/output_npy_01'
    time1 = time.time()
    gen_results(in_dir)
    time2=time.time()
    print ('time use: ' + str(time2 - time1) + ' s')