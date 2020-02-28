# -*- coding: utf-8 -*-
"""
haikou traffice
fun:
	shape2hotmap
env:
	Win7 64bit anaconda;python 3.5;tensorflow1.10.1;Keras2.2.4
	pip2,matplotlib2.2.3,seaborn0.9.0
"""
from __future__ import print_function
import os
import sys
import numpy as np
import shutil
import time
import seaborn as sns
import numpy as np
import pandas as pd
import folium
import folium.plugins as plugins
import webbrowser
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import rcParams
import re
import h5py
import time
#sys.setrecursionlimit(10000000)#手动设置递归调用深度

def shape2hotmap(in_dir,out_dir):
    fr = h5py.File(in_dir, 'r')
    print("reading ", in_dir)
    # print("fr.keys() = ", fr.keys()) # python2
    print("fr.keys() = ",[key for key in fr.keys()])  # python3
    key_list = [key for key in fr.keys()]
    print('num of key_list =', len(key_list))
    #key_list.sort()
    for key in key_list:  # data.shape[0] = 48
        print("------------------",key,"------------------")
        file_path = os.path.join(out_dir, key)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        data = fr[key][()]# fr[key].value  dataset.value has been deprecated.
        print("data.shape = ", data.shape) # (24, 2, 75, 50)
        #print("data.dtype = ", data.dtype) # int32
        #print("max data = ", np.max(data)) # max data =  415
        #print("min data = ", np.min(data)) # min data =  0
        #data = data / np.max(data)  # 归一化，（0-1），最大值为1
        data = data.transpose((0, 1, 3, 2))
        #data = data.swapaxes(2, 3)
        print("data.shape after swap", data.shape)  # (24, 2, 75, 50)
        for i in range(data.shape[0]) :
            plt.subplots()
            sns.heatmap(data[i, 0, :, :], vmin=0, vmax=200,
                        annot = False,cbar = False,square = True,#是否注释数值，绘制图条，内部是否是方格 当annot为True时，在heatmap中每个方格写入数据 annot_kws，当annot为True时，可设置各个参数，包括大小，颜色，加粗，斜体字等
                        #cmap="YlGnBu",#颜色方案 默认为cubehelix map (数据集为连续数据集时) 或 RdBu_r (数据集为离散数据集时) 'rainbow'
                        xticklabels=5, yticklabels=5,#X Y轴刻度间隔，False不绘制刻度标签
                        linewidths=0)#内部方格间距
                        # np.max(data) # elesun shape
            # save png
            save_path = os.path.join(file_path, "start_%02d.png"%(i))
            plt.savefig(save_path)#  bbox_inches='tight' dpi=300
            #plt.close()

            plt.subplots()
            sns.heatmap(data[i, 1, :, :], vmin=0, vmax=200,
                        annot=False, cbar=False, square=True,
                        # 是否注释数值，绘制图条，内部是否是方格 当annot为True时，在heatmap中每个方格写入数据 annot_kws，当annot为True时，可设置各个参数，包括大小，颜色，加粗，斜体字等
                        # cmap="YlGnBu",#颜色方案 默认为cubehelix map (数据集为连续数据集时) 或 RdBu_r (数据集为离散数据集时) 'rainbow'
                        xticklabels=5, yticklabels=5,  # X Y轴刻度间隔，False不绘制刻度标签 auto
                        linewidths=0)  # 内部方格间距
                        # np.max(data) # elesun shape
            # save png
            save_path = os.path.join(file_path, "dest_%02d.png" % (i))
            plt.savefig(save_path)
            # plt.show()
            plt.close("all")

    fr.close()
    print("closed h5 and hotmap generated!")


if __name__ == '__main__':
    in_dir = "stage_03/haikou_shape.h5"
    out_dir = "stage_04"
    if not os.path.exists(in_dir):
        print("there is not directory", in_dir)
    print("input directory is ", in_dir)
    if os.path.exists(out_dir):
        # os.remove(out_dir) #删除整个目录
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    time1 = time.time()
    shape2hotmap(in_dir, out_dir)
    time2=time.time()
    print ('time use: ' + str(time2 - time1) + ' s')